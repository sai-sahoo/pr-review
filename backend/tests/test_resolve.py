"""Step 15b-2: the bot resolves its own threads once the code they point at is gone."""

import json

import httpx
import pytest

from app.fingerprint import code_fingerprint
from app.github_client import GitHubError, ReviewThread, get_open_bot_threads, resolve_thread
from app.graph.nodes import resolve_fixed
from app.schemas import SkippedFile
from sample_pr import PATH, PAYMENTS_PY, make_pr

# --- talking to GitHub's GraphQL API --------------------------------------


@pytest.fixture
def fake_graphql(monkeypatch):
    """Answer POST /graphql with the replies queued in the returned list; the
    requests sent are collected in the other.
    """
    sent: list[dict] = []
    replies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/graphql"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=replies.pop(0))

    monkeypatch.setattr("app.github_client._app_client", lambda token: httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)))
    return sent, replies


def thread(id: str, body: str, author: str = "Bot", resolved: bool = False) -> dict:
    return {"id": id, "isResolved": resolved, "path": PATH,
            "comments": {"nodes": [{"body": body, "author": {"__typename": author}}]}}


def page(threads: list[dict], next_cursor: str | None = None) -> dict:
    return {"data": {"repository": {"pullRequest": {"reviewThreads": {
        "pageInfo": {"hasNextPage": next_cursor is not None, "endCursor": next_cursor},
        "nodes": threads,
    }}}}}


def test_reads_only_our_open_threads_on_every_page(fake_graphql):
    sent, replies = fake_graphql
    replies += [
        page([
            thread("T1", "bug <!-- pr-review:aaaaaaaaaaaa -->"),
            thread("T2", "bug <!-- pr-review:bbbbbbbbbbbb -->", resolved=True),  # already resolved
            thread("T3", "I think <!-- pr-review:cccccccccccc -->", author="User"),  # a human started it
            thread("T4", "another bot, no marker"),
        ], next_cursor="c1"),
        page([thread("T5", "<!-- pr-review:dddddddddddd -->")]),
    ]

    threads = get_open_bot_threads(make_pr(), "ghs_abc")

    assert [(t.id, t.fingerprint) for t in threads] == [("T1", "aaaaaaaaaaaa"), ("T5", "dddddddddddd")]
    assert [s["variables"]["after"] for s in sent] == [None, "c1"]  # the cursor fetched page 2


def test_graphql_errors_arrive_with_status_200_and_still_raise(fake_graphql):
    _, replies = fake_graphql
    replies.append({"data": None, "errors": [{"message": "Resource not accessible by integration"}]})

    with pytest.raises(GitHubError, match="Resource not accessible by integration"):
        resolve_thread("ghs_abc", "T1")


def test_resolve_sends_the_thread_id(fake_graphql):
    sent, replies = fake_graphql
    replies.append({"data": {"resolveReviewThread": {"thread": {"isResolved": True}}}})

    resolve_thread("ghs_abc", "T1")

    assert "resolveReviewThread" in sent[0]["query"] and sent[0]["variables"] == {"id": "T1"}


# --- the resolve_fixed node --------------------------------------------------


@pytest.fixture
def app_installed(monkeypatch):
    monkeypatch.setenv("GITHUB_APP_ID", "1")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "unused.pem")
    monkeypatch.setattr("app.graph.nodes.installation_token", lambda owner, repo: "ghs_abc")


def on(path: str, code: str, id: str) -> ReviewThread:
    """An open bot thread about this line of code."""
    return ReviewThread(id=id, path=path, fingerprint=code_fingerprint(path, code))


def test_resolves_only_threads_whose_line_is_gone(fake_github, app_installed):
    pr = make_pr()
    pr.skipped.append(SkippedFile(path="app/old.py", reason="deleted"))
    resolved = fake_github(pr, files={PATH: PAYMENTS_PY}, threads=[
        on(PATH, "order.balance -= amount", "still-there"),  # unchanged (wherever it sits now)
        on(PATH, "if amount > order.total:", "edited"),  # no longer in the file
        on("app/old.py", "x = 1", "file-deleted"),
    ])

    update = resolve_fixed({"pr": pr})

    assert resolved == ["edited", "file-deleted"]
    assert update == {"resolved": [PATH, "app/old.py"], "resolve_note": "resolved 2 fixed threads"}


def test_a_file_that_cannot_be_read_keeps_its_threads_open(fake_github, app_installed):
    resolved = fake_github(make_pr(), files={}, threads=[on(PATH, "if amount > order.total:", "T1")])

    assert resolve_fixed({"pr": make_pr()})["resolved"] == []
    assert resolved == []


def test_github_refusing_does_not_fail_the_review(fake_github, app_installed, monkeypatch):
    fake_github(make_pr())

    def refused(pr, token):
        raise GitHubError("GitHub GraphQL error: Resource not accessible by integration")

    monkeypatch.setattr("app.graph.nodes.get_open_bot_threads", refused)

    update = resolve_fixed({"pr": make_pr()})  # returns instead of raising

    assert update["resolved"] == []
    assert update["resolve_note"].startswith("could not resolve threads: GitHub GraphQL error")


def test_without_an_app_nothing_is_asked():
    # conftest leaves the App unconfigured; the seams would fail the test if called.
    assert resolve_fixed({"pr": make_pr()}) == {"resolved": [], "resolve_note": "skipped: no GitHub App configured"}
