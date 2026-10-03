"""The HTTP layer: status codes, the job lifecycle, and error handling.

TestClient waits for background tasks before returning, so by the time
POST comes back the review has already run. The real server would still be
"running" at that moment; here we can check the end state deterministically.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.github_client import GitHubError
from app.graph.prompts import SPECIALIST_PROMPTS, TRIAGE_PROMPT, VERIFIER_PROMPT
from fakes import by_system_prompt, scripted, tool_call
from sample_pr import finding, make_pr

PR_URL = "https://github.com/acme/shop/pull/7"


@pytest.fixture
def client():
    return TestClient(create_app())  # a fresh store per test


def test_review_lifecycle(client, fake_llm, fake_github):
    fake_github(make_pr())
    bug = finding(line=11, severity="critical")
    fake_llm(by_system_prompt({
        TRIAGE_PROMPT: scripted(tool_call("TriagePlan", reason="r", specialists=["correctness"])),
        SPECIALIST_PROMPTS["correctness"]: scripted(tool_call("Review", findings=[bug.model_dump()])),
        VERIFIER_PROMPT: scripted(tool_call("VerifierReport", verdicts=[
            {"id": 0, "reason": "real", "confidence": 0.9},
        ])),
    }))

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
