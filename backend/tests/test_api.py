"""The HTTP layer and the worker: status codes, the job lifecycle, and error handling.

POST only queues a job, so tests call run_worker() afterwards: a real arq
worker, in burst mode, runs every queued review and then stops. Queue and
pub/sub go through fakeredis, an in-memory Redis, on the app's event loop.

Each test gets its own SQLite file instead of Postgres: no Docker needed, and
nothing leaks between tests. The tables come from Base.metadata.create_all,
the quick way; the real database gets them from Alembic migrations.
"""

import hashlib
import hmac
import json

import pytest
from arq import ArqRedis
from arq.worker import create_worker
from fakeredis import FakeServer
from fakeredis.aioredis import FakeAsyncRedisConnection
from fastapi.testclient import TestClient
from redis.asyncio import ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

from app.api.main import create_app
from app.db import Base
from app.github_client import GitHubError
from app.graph import build_graph
from app.graph.checkpointer import open_checkpointer
from app.graph.prompts import SPECIALIST_PROMPTS, TRIAGE_PROMPT, VERIFIER_PROMPT
from fakes import by_system_prompt, scripted, tool_call
from app.worker import WorkerSettings, enqueue_review, shutdown
from sample_pr import PATH, PAYMENTS_PY, finding, make_pr

PR_URL = "https://github.com/acme/shop/pull/7"
SHA = "a" * 40
SECRET = "test-webhook-secret"


@pytest.fixture
def db_url(tmp_path) -> str:
    """A fresh database file for this test, with the tables already created."""
    path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{path}")  # a plain sync engine is enough here
    Base.metadata.create_all(engine)
    engine.dispose()
    return f"sqlite+aiosqlite:///{path}"  # what the async app uses


@pytest.fixture(autouse=True)
def quiet_arq(monkeypatch):
    """At startup arq logs the server's INFO, a command fakeredis doesn't have."""

    async def skip(*args):
        pass

    monkeypatch.setattr("arq.worker.log_redis_info", skip)


@pytest.fixture(autouse=True)
def pr_head(monkeypatch):
    """POST asks GitHub for the PR's head commit. By default every PR is at
    SHA, under its canonical URL. A test moves the PR to a new commit with
    pr_head["sha"] = "...".
    """
    head = {"sha": SHA}
    monkeypatch.setattr("app.api.main.get_pr_head", lambda url: (PR_URL, head["sha"]))
    return head


def fake_redis() -> ArqRedis:
    """An arq client whose connections talk to an in-memory FakeServer."""
    return ArqRedis(connection_pool=ConnectionPool(connection_class=FakeAsyncRedisConnection, server=FakeServer()))


@pytest.fixture
def client(db_url):
    # `with` runs the app's startup and shutdown, and keeps one event loop for
    # the whole test: the worker and the SSE handlers run on it too.
    with TestClient(create_app(db_url, fake_redis(), webhook_secret=SECRET)) as client:
        yield client


async def burst_worker(app) -> None:
    """A real arq worker built from our WorkerSettings, on the app's Redis and
    database. burst=True: run everything queued, then return instead of
    waiting for more.
    """
    worker = create_worker(
        WorkerSettings,
        redis_pool=app.state.redis,
        ctx={"db_url": app.state.db_url},
        burst=True,
        handle_signals=False,  # signal handlers only work in the main thread
        poll_delay=0.01,  # check the queue every 10 ms instead of 0.5 s
    )
    await worker.async_run()
    # async_run skips on_shutdown. (worker.close() would run it, but would
    # also close the app's Redis client.)
    await shutdown(worker.ctx)


def run_worker(client) -> None:
    """Block until the worker has run every queued review. portal.call runs a
    coroutine on the client's event loop.
    """
    client.portal.call(burst_worker, client.app)


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
    assert resp.json()["status"] == "queued"
    review_id = resp.json()["id"]

    # The API did no work: the review is a job waiting in Redis, named by its id.
    assert client.get(f"/reviews/{review_id}").json()["status"] == "queued"
    jobs = client.portal.call(client.app.state.redis.queued_jobs)
    assert [(j.function, j.job_id) for j in jobs] == [("run_review", review_id)]

    run_worker(client)

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
    run_worker(client)
    body = client.get(f"/reviews/{review_id}").json()

    assert body["status"] == "failed"
    assert body["error"] == "PR not found."


def test_crash_marks_review_failed_instead_of_stuck_running(client):
    # No fakes installed: the conftest safety net raises AssertionError inside
    # the graph, standing in for any unexpected bug.
    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]
    run_worker(client)
    body = client.get(f"/reviews/{review_id}").json()

    assert body["status"] == "failed"
    assert body["error"].startswith("internal error:")


def test_event_stream_tells_the_whole_story_in_order(client, fake_llm, fake_github):
    script_one_bug(fake_llm, fake_github)
    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]

    # Worker in the background, stream in the foreground: the SSE handler
    # wakes up on Redis messages while the review is still running.
    worker = client.portal.start_task_soon(burst_worker, client.app)
    events = read_sse(client, review_id)
    worker.result()

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
        ("node", "publish"),
        ("done", None),
    ]
    data = [d for _, _, d in events]
    assert data[3] == {"type": "agent", "agent": "correctness", "tool": "read_file",
                       "args": {"path": PATH, "start_line": 1, "end_line": 20}}
    assert data[4] == {"type": "agent", "agent": "correctness", "submitted": 1}
    assert [f["line"] for f in data[5]["findings"]] == [11]
    assert data[8] == {"type": "node", "node": "publish", "review_url": None,
                       "note": "not posted: no GitHub App configured"}
    assert data[-1] == {"type": "done", "kept": 1}


def test_reconnect_resumes_after_last_event_id(client, fake_llm, fake_github):
    script_one_bug(fake_llm, fake_github)
    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]
    run_worker(client)
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
    run_worker(client)

    events = read_sse(client, review_id)

    assert [name for _, name, _ in events] == ["status", "failed"]
    assert events[-1][2] == {"type": "failed", "error": "PR not found."}


def test_events_for_unknown_id_is_a_404(client):
    assert client.get("/reviews/does-not-exist/events").status_code == 404


def test_reviews_survive_a_restart(db_url, fake_llm, fake_github):
    script_one_bug(fake_llm, fake_github)
    with TestClient(create_app(db_url, fake_redis())) as client:
        review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]
        run_worker(client)
        before = read_sse(client, review_id)

    # A brand-new app and Redis: nothing left in memory, only what's in the database.
    with TestClient(create_app(db_url, fake_redis())) as client:
        body = client.get(f"/reviews/{review_id}").json()
        assert body["status"] == "done"
        assert [f["line"] for f in body["findings"]] == [11]
        assert read_sse(client, review_id) == before  # the UI can replay it all


def test_worker_resumes_an_interrupted_review(db_url, fake_llm, fake_github, monkeypatch):
    script_one_bug(fake_llm, fake_github)  # scripted: each agent may answer only once
    fetches = []
    monkeypatch.setattr("app.graph.nodes.get_pull_request", lambda url: fetches.append(url) or make_pr())

    with TestClient(create_app(db_url, fake_redis())) as client:
        store = client.app.state.store
        record = client.portal.call(store.create, PR_URL)

        async def crash_after_triage():
            # Stand-in for a worker killed mid-review: the record says
            # "running", and the checkpoints end right after triage.
            await store.update(record.id, status="running")
            async with open_checkpointer(db_url) as checkpointer:
                graph = build_graph(checkpointer)
                config = {"configurable": {"thread_id": record.id}}
                async for _ in graph.astream({"pr_url": PR_URL}, config, interrupt_after=["triage"]):
                    pass

        client.portal.call(crash_after_triage)
        # Nothing was ever queued: as if Redis had lost its data too. The
        # worker's startup finds the review in Postgres and queues it again.
        run_worker(client)
        body = client.get(f"/reviews/{record.id}").json()
        events = read_sse(client, record.id)

    assert body["status"] == "done"
    assert body["title"] == make_pr().title  # from the checkpoint: fetch_pr didn't run again
    assert [f["line"] for f in body["findings"]] == [11]
    assert fetches == [PR_URL]  # fetched once, before the "crash"
    # Triage didn't run again either: its script had only one answer, so a
    # second call would have failed the review.
    assert events[0][2] == {"type": "status", "status": "running", "resumed": True}
    assert [d.get("node") for _, name, d in events if name == "node"] == [
        "specialist", "aggregate", "verify", "publish"]


def test_a_finished_review_is_not_run_twice(client, fake_llm, fake_github):
    script_one_bug(fake_llm, fake_github)  # each agent may answer only once
    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]
    run_worker(client)
    before = read_sse(client, review_id)

    # The same job delivered again. The job sees "done" and returns at once;
    # running the graph again would have hit the empty scripts and failed.
    client.portal.call(enqueue_review, client.app.state.redis, review_id)
    run_worker(client)

    assert client.get(f"/reviews/{review_id}").json()["status"] == "done"
    assert read_sse(client, review_id) == before  # no new events


def test_queue_down_is_a_503_and_the_review_is_failed(client, monkeypatch):
    async def refused(*args, **kwargs):
        raise RedisConnectionError("Connection refused")

    monkeypatch.setattr(client.app.state.redis, "enqueue_job", refused)

    resp = client.post("/reviews", json={"pr_url": PR_URL})

    assert resp.status_code == 503
    [record] = client.get("/reviews").json()  # saved, but not left "queued" forever
    assert record["status"] == "failed"
    assert record["error"] == "queue unavailable"


# --- dedupe by (PR, head commit) -----------------------------------------


def test_same_pr_and_commit_returns_the_existing_review(client):
    first = client.post("/reviews", json={"pr_url": PR_URL})
    # Another spelling of the same PR: the canonical URL from GitHub makes them one key.
    again = client.post("/reviews", json={"pr_url": PR_URL + "/files"})

    assert first.status_code == 202
    assert again.status_code == 200  # nothing new to do
    assert again.json()["id"] == first.json()["id"]
    assert first.json()["head_sha"] == SHA
    assert len(client.get("/reviews").json()) == 1
    jobs = client.portal.call(client.app.state.redis.queued_jobs)
    assert len(jobs) == 1  # and only one job: no second LLM bill


def test_done_review_is_returned_too(client, fake_llm, fake_github):
    script_one_bug(fake_llm, fake_github)  # scripted: each agent may answer only once
    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]
    run_worker(client)

    resp = client.post("/reviews", json={"pr_url": PR_URL})

    assert resp.status_code == 200
    assert resp.json()["id"] == review_id
    assert resp.json()["status"] == "done"
    assert [f["line"] for f in resp.json()["findings"]] == [11]


def test_a_new_commit_gets_a_new_review(client, pr_head):
    first = client.post("/reviews", json={"pr_url": PR_URL}).json()
    pr_head["sha"] = "b" * 40  # someone pushed to the PR

    second = client.post("/reviews", json={"pr_url": PR_URL})

    assert second.status_code == 202
    assert second.json()["id"] != first["id"]
    assert second.json()["head_sha"] == "b" * 40


def test_a_failed_review_can_be_retried(client, monkeypatch):
    def not_found(url):
        raise GitHubError("PR not found.")

    monkeypatch.setattr("app.graph.nodes.get_pull_request", not_found)
    failed_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]
    run_worker(client)
    assert client.get(f"/reviews/{failed_id}").json()["status"] == "failed"

    retry = client.post("/reviews", json={"pr_url": PR_URL})

    assert retry.status_code == 202
    assert retry.json()["id"] != failed_id


def test_unknown_pr_is_rejected_before_anything_is_saved(client, monkeypatch):
    def not_found(url):
        raise GitHubError("PR not found. Check the URL; private repos need GITHUB_TOKEN.")

    monkeypatch.setattr("app.api.main.get_pr_head", not_found)

    resp = client.post("/reviews", json={"pr_url": PR_URL})

    assert resp.status_code == 422
    assert resp.json()["detail"].startswith("PR not found.")
    assert client.get("/reviews").json() == []


def test_database_rejects_a_second_live_review_of_one_commit(client):
    store = client.app.state.store
    first = client.portal.call(store.create, PR_URL, SHA)

    # Bypassing get_or_create, as a racing request would: the index says no.
    with pytest.raises(IntegrityError):
        client.portal.call(store.create, PR_URL, SHA)

    # Once the first one has failed, it no longer counts.
    client.portal.call(lambda: store.update(first.id, status="failed"))
    client.portal.call(store.create, PR_URL, SHA)


def test_losing_the_race_returns_the_winner(client, monkeypatch):
    store = client.app.state.store
    winner = client.portal.call(store.create, PR_URL, SHA)
    # Pretend the winner's INSERT landed just after our SELECT: the first
    # lookup sees nothing, so we try to insert and hit the unique index.
    real_find = store.find_live
    calls = []

    async def find_live_late(*args):
        calls.append(args)
        return None if len(calls) == 1 else await real_find(*args)

    monkeypatch.setattr(store, "find_live", find_live_late)

    record, created = client.portal.call(store.get_or_create, PR_URL, SHA)

    assert (record.id, created) == (winner.id, False)
    assert len(calls) == 2


# --- GitHub webhook --------------------------------------------------------


def pr_event(action="opened", sha=SHA, draft=False) -> dict:
    """The few fields of a pull_request payload we read (the real one has hundreds)."""
    return {"action": action, "pull_request": {"html_url": PR_URL, "draft": draft, "head": {"sha": sha}}}


def deliver(client, payload: dict, event="pull_request", secret=SECRET, signature=None):
    """POST the way GitHub does: JSON body, event name, and the HMAC of the exact bytes."""
    body = json.dumps(payload).encode()
    if signature is None:
        signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post("/webhooks/github", content=body, headers={
        "Content-Type": "application/json",
        "X-GitHub-Event": event,
        "X-GitHub-Delivery": "d-1",
        "X-Hub-Signature-256": signature,
    })


def test_opened_pr_queues_a_review_without_asking_github(client, monkeypatch):
    def refuse(url):
        raise AssertionError("the payload already has the head sha")

    monkeypatch.setattr("app.api.main.get_pr_head", refuse)

    resp = deliver(client, pr_event(sha="c" * 40))

    assert resp.status_code == 202
    [record] = client.get("/reviews").json()
    assert resp.json() == {"review_id": record["id"], "created": True}
    assert (record["pr_url"], record["head_sha"], record["status"]) == (PR_URL, "c" * 40, "queued")
    jobs = client.portal.call(client.app.state.redis.queued_jobs)
    assert [j.job_id for j in jobs] == [record["id"]]


@pytest.mark.parametrize("signature", [
    "",  # missing
    "sha256=" + "0" * 64,  # well-formed, wrong
    "sha256=café".encode(),  # raw non-ASCII bytes; must be a 401, not a crash
])
def test_bad_signature_is_a_401(client, signature):
    resp = deliver(client, pr_event(), signature=signature)

    assert resp.status_code == 401
    assert client.get("/reviews").json() == []


def test_signed_with_another_secret_is_a_401(client):
    assert deliver(client, pr_event(), secret="guessed").status_code == 401


def test_one_changed_byte_breaks_the_signature(client):
    body = json.dumps(pr_event()).encode()
    signature = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    tampered = body.replace(SHA.encode(), b"b" * 40)  # someone edits the payload in transit

    resp = client.post("/webhooks/github", content=tampered, headers={
        "X-GitHub-Event": "pull_request", "X-Hub-Signature-256": signature})

    assert resp.status_code == 401


def test_redelivery_returns_the_same_review(client):
    first = deliver(client, pr_event())
    again = deliver(client, pr_event())  # GitHub retried, or you clicked "Redeliver"
    from_ui = client.post("/reviews", json={"pr_url": PR_URL})  # and someone pasted the URL

    assert (first.status_code, again.status_code, from_ui.status_code) == (202, 200, 200)
    assert again.json()["review_id"] == first.json()["review_id"] == from_ui.json()["id"]
    assert len(client.get("/reviews").json()) == 1


def test_push_to_the_pr_starts_a_new_review(client):
    first = deliver(client, pr_event()).json()["review_id"]

    resp = deliver(client, pr_event(action="synchronize", sha="b" * 40))

    assert resp.status_code == 202
    assert resp.json()["review_id"] != first


@pytest.mark.parametrize(("event", "payload", "reason"), [
    ("ping", {"zen": "Keep it logically awesome."}, None),
    ("issues", {"action": "opened"}, "event 'issues'"),
    ("pull_request", pr_event(action="closed"), "action 'closed'"),
    ("pull_request", pr_event(action="labeled"), "action 'labeled'"),
    ("pull_request", pr_event(draft=True), "draft PR"),
])
def test_other_deliveries_are_acknowledged_and_ignored(client, event, payload, reason):
    resp = deliver(client, payload, event=event)

    assert resp.status_code == 200  # a 2xx, so GitHub doesn't mark it failed
    assert resp.json() == ({"ok": "pong"} if reason is None else {"ignored": reason})
    assert client.get("/reviews").json() == []


def test_draft_marked_ready_starts_a_review(client):
    assert deliver(client, pr_event(action="ready_for_review")).status_code == 202


def test_signed_but_unexpected_payload_is_a_422(client):
    assert deliver(client, {"action": "opened"}).status_code == 422


def test_webhooks_are_off_without_a_secret(db_url):
    # conftest sets GITHUB_WEBHOOK_SECRET to "": an app without a secret
    # refuses every delivery, instead of trusting unsigned ones.
    with TestClient(create_app(db_url, fake_redis())) as client:
        resp = deliver(client, pr_event(), secret="")

    assert resp.status_code == 503


# --- human approval ----------------------------------------------------------


@pytest.fixture
def installed_app(monkeypatch):
    """The App is configured and installed on the PR's repo, so reviews pause
    for approval before posting. Returns the payloads that got posted.
    """
    monkeypatch.setenv("GITHUB_APP_ID", "1")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "unused.pem")  # installation_token is faked
    monkeypatch.setattr("app.graph.nodes.installation_token", lambda owner, repo: "ghs_abc")
    posted: list[dict] = []
    monkeypatch.setattr("app.graph.nodes.post_review",
                        lambda pr, payload, token: posted.append(payload) or "https://github.com/r/1")
    return posted


def waiting_review(client, fake_llm, fake_github) -> str:
    """A review that has run up to the approval step, with one finding (line 11)."""
    script_one_bug(fake_llm, fake_github)
    review_id = client.post("/reviews", json={"pr_url": PR_URL}).json()["id"]
    run_worker(client)
    return review_id


def test_review_waits_for_approval_then_posts(client, fake_llm, fake_github, installed_app):
    review_id = waiting_review(client, fake_llm, fake_github)

    # The job has ended: the review is parked, with its findings ready to judge.
    body = client.get(f"/reviews/{review_id}").json()
    assert body["status"] == "waiting"
    assert [f["line"] for f in body["findings"]] == [11]
    assert installed_app == []  # nothing posted yet
    assert client.portal.call(client.app.state.redis.queued_jobs) == []  # and no job holds a slot

    resp = client.post(f"/reviews/{review_id}/approval", json={"approved": [0]})
    assert resp.status_code == 202
    assert (resp.json()["status"], resp.json()["approved"]) == ("queued", [0])

    run_worker(client)  # the resume: scripted agents have no answers left, so none may run again

    body = client.get(f"/reviews/{review_id}").json()
    assert body["status"] == "done"
    [payload] = installed_app
    assert [c["line"] for c in payload["comments"]] == [11]
    events = [(name, d.get("node") or d.get("status")) for _, name, d in read_sse(client, review_id)]
    assert events[events.index(("node", "verify")):] == [
        ("node", "verify"),
        ("status", "waiting"),  # the first job ends here
        ("status", "running"),  # the second one starts, from the checkpoint
        ("node", "approve"),
        ("node", "publish"),
        ("done", None),
    ]


def test_dismissing_every_finding_posts_nothing(client, fake_llm, fake_github, installed_app):
    review_id = waiting_review(client, fake_llm, fake_github)

    client.post(f"/reviews/{review_id}/approval", json={"approved": []})
    run_worker(client)

    assert client.get(f"/reviews/{review_id}").json()["status"] == "done"
    assert installed_app == []
    publish = [d for _, _, d in read_sse(client, review_id) if d.get("node") == "publish"]
    assert publish[0]["note"] == "not posted: every finding was dismissed"


def test_approval_is_only_accepted_while_waiting(client):
    queued_id = client.post("/reviews", json={"pr_url": "https://github.com/acme/other/pull/1"}).json()["id"]
    assert client.post(f"/reviews/{queued_id}/approval", json={"approved": []}).status_code == 409
    assert client.post("/reviews/nope/approval", json={"approved": []}).status_code == 404


def test_approval_positions_are_checked(client, fake_llm, fake_github, installed_app):
    review_id = waiting_review(client, fake_llm, fake_github)

    resp = client.post(f"/reviews/{review_id}/approval", json={"approved": [1]})  # there's only [0]

    assert resp.status_code == 422
    assert client.get(f"/reviews/{review_id}").json()["status"] == "waiting"  # still undecided


def test_a_second_decision_is_a_409(client, fake_llm, fake_github, installed_app):
    review_id = waiting_review(client, fake_llm, fake_github)
    store = client.app.state.store

    assert client.portal.call(store.approve, review_id, [0]) is True
    # A second click that got past the status check at the same moment: the
    # conditional UPDATE matches nothing, because the status already changed.
    assert client.portal.call(store.approve, review_id, []) is False
    assert client.get(f"/reviews/{review_id}").json()["approved"] == [0]


def test_a_waiting_review_is_not_started_by_a_stray_job(client, fake_llm, fake_github, installed_app):
    review_id = waiting_review(client, fake_llm, fake_github)

    client.portal.call(enqueue_review, client.app.state.redis, review_id)  # e.g. a redelivery
    run_worker(client)

    assert client.get(f"/reviews/{review_id}").json()["status"] == "waiting"
    assert installed_app == []


def test_queue_down_on_approval_keeps_the_review_waiting(client, fake_llm, fake_github, installed_app,
                                                       monkeypatch):
    review_id = waiting_review(client, fake_llm, fake_github)

    async def refused(*args, **kwargs):
        raise RedisConnectionError("Connection refused")

    monkeypatch.setattr(client.app.state.redis, "enqueue_job", refused)

    resp = client.post(f"/reviews/{review_id}/approval", json={"approved": [0]})

    assert resp.status_code == 503
    body = client.get(f"/reviews/{review_id}").json()
    assert (body["status"], body["approved"]) == ("waiting", None)  # free to send it again
