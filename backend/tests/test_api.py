"""The HTTP layer: status codes, the job lifecycle, and error handling.

TestClient waits for background tasks before returning, so by the time
POST comes back the review has already run. The real server would still be
"running" at that moment; here we can check the end state deterministically.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.github_client import GitHubError
from app.graph.prompts import SPECIALIST_PROMPTS, TRIAGE_PROMPT, VERIFIER_PROMPT
from fakes import by_system_prompt, scripted, tool_call
from sample_pr import PATH, PAYMENTS_PY, finding, make_pr

PR_URL = "https://github.com/acme/shop/pull/7"


@pytest.fixture
def client():
    # `with` keeps one event loop for the whole test. Without it, each request
    # gets its own loop, and the store's asyncio.Condition must stay on one.
    with TestClient(create_app()) as client:  # a fresh store per test
        yield client


def read_sse(client, review_id, **headers) -> list[tuple[str, str, dict]]:
    """GET the event stream and parse it into (id, event, data) triples."""
    resp = client.get(f"/reviews/{review_id}/events", headers=headers)
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = []
    for block in filter(None, resp.text.strip().split("\n\n")):  # a blank line ends each event
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["id"], fields["event"], json.loads(fields["data"])))
    return events


def script_one_bug(fake_llm, fake_github):
    """A full run: the correctness agent reads a file, then reports one bug."""
    fake_github(make_pr(), files={PATH: PAYMENTS_PY})
    fake_llm(by_system_prompt({
        TRIAGE_PROMPT: scripted(tool_call("TriagePlan", reason="r", specialists=["correctness"])),
        SPECIALIST_PROMPTS["correctness"]: scripted(
            tool_call("read_file", path=PATH, start_line=1, end_line=20),
            tool_call("Review", findings=[finding(line=11, severity="critical").model_dump()]),
        ),
        VERIFIER_PROMPT: scripted(tool_call("VerifierReport", verdicts=[
            {"id": 0, "reason": "real", "confidence": 0.9},
        ])),
    }))


def test_review_lifecycle(client, fake_llm, fake_github):
    script_one_bug(fake_llm, fake_github)

    resp = client.post("/reviews", json={"pr_url": PR_URL})
    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"  # the response was built before the job ran
    review_id = resp.json()["id"]

    body = client.get(f"/reviews/{review_id}").json()
    assert body["status"] == "done"
    assert body["title"] == make_pr().title
    assert [f["line"] for f in body["findings"]] == [11]
    assert body["checks"][0]["kept"] is True
    assert body["finished_at"] is not None

    assert [r["id"] for r in client.get("/reviews").json()] == [review_id]


def test_bad_url_is_rejected_before_any_work(client):
    resp = client.post("/reviews", json={"pr_url": "https://example.com/nope"})
    assert resp.status_code == 422
    assert client.get("/reviews").json() == []  # nothing was queued


def test_missing_body_field_is_a_422(client):
    assert client.post("/reviews", json={}).status_code == 422  # Pydantic validation


def test_cors_lets_the_frontend_call_the_api(client):
    # Before a JSON POST from another origin, the browser asks with OPTIONS.
    resp = client.options("/reviews", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"

    # Any other origin gets no allow header, so the browser blocks it.
    resp = client.get("/reviews", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp.headers


def test_unknown_id_is_a_404(client):
    assert client.get("/reviews/does-not-exist").status_code == 404


def test_github_error_marks_review_failed(client, monkeypatch):
    def not_found(url):
        raise GitHubError("PR not found.")

    monkeypatch.setattr("app.graph.nodes.get_pull_request", not_found)

    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]
    body = client.get(f"/reviews/{review_id}").json()

    assert body["status"] == "failed"
    assert body["error"] == "PR not found."


def test_crash_marks_review_failed_instead_of_stuck_running(client):
    # No fakes installed: the conftest safety net raises AssertionError inside
    # the graph, standing in for any unexpected bug.
    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]
    body = client.get(f"/reviews/{review_id}").json()

    assert body["status"] == "failed"
    assert body["error"].startswith("internal error:")


def test_event_stream_tells_the_whole_story_in_order(client, fake_llm, fake_github):
    script_one_bug(fake_llm, fake_github)
    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]

    events = read_sse(client, review_id)

    assert [i for i, _, _ in events] == [str(n) for n in range(len(events))]
    assert [(name, data.get("node") or data.get("status")) for _, name, data in events] == [
        ("status", "running"),
        ("node", "fetch_pr"),
        ("node", "triage"),
        ("agent", None),  # read_file, sent while the specialist was still running
        ("agent", None),  # submitted
        ("node", "specialist"),
        ("node", "aggregate"),
        ("node", "verify"),
        ("done", None),
    ]
    data = [d for _, _, d in events]
    assert data[3] == {"type": "agent", "agent": "correctness", "tool": "read_file",
                       "args": {"path": PATH, "start_line": 1, "end_line": 20}}
    assert data[4] == {"type": "agent", "agent": "correctness", "submitted": 1}
    assert [f["line"] for f in data[5]["findings"]] == [11]
    assert data[-1] == {"type": "done", "kept": 1}


def test_reconnect_resumes_after_last_event_id(client, fake_llm, fake_github):
    script_one_bug(fake_llm, fake_github)
    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]
    everything = read_sse(client, review_id)

    resumed = read_sse(client, review_id, **{"Last-Event-ID": "5"})

    assert resumed == everything[6:]
    # Already past the final event: the stream ends at once instead of hanging.
    assert read_sse(client, review_id, **{"Last-Event-ID": everything[-1][0]}) == []


def test_failed_review_stream_ends_with_failed(client, monkeypatch):
    def not_found(url):
        raise GitHubError("PR not found.")

    monkeypatch.setattr("app.graph.nodes.get_pull_request", not_found)
    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]

    events = read_sse(client, review_id)

    assert [name for _, name, _ in events] == ["status", "failed"]
    assert events[-1][2] == {"type": "failed", "error": "PR not found."}


def test_events_for_unknown_id_is_a_404(client):
    assert client.get("/reviews/does-not-exist/events").status_code == 404
