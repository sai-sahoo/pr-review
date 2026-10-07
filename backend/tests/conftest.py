"""Shared fixtures. pytest loads this file automatically before any test.

Patch where a name is *looked up*, not where it is defined: nodes.py did
`from app.llm import get_model`, so it holds its own reference, and patching
app.llm.get_model would not touch it.
"""

import os

# Before any `import app`: app/__init__ loads .env but never overrides a
# variable that's already set. So the module-level app in app.api.main gets a
# throwaway in-memory SQLite database, and no test can touch your Postgres.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"
# Tests pass a fake Redis. If one ever forgot, this port has nothing listening:
# a job must never reach your real Redis, where a real worker would run it
# with real LLM calls.
os.environ["REDIS_URL"] = "redis://localhost:1"
# Webhooks off unless a test passes its own secret: your real secret from
# .env never takes part in a test.
os.environ["GITHUB_WEBHOOK_SECRET"] = ""
# Same for the App: the publish node sees "not configured" and posts nothing.
# Tests that cover posting set these themselves.
os.environ["GITHUB_APP_ID"] = ""
os.environ["GITHUB_APP_PRIVATE_KEY_PATH"] = ""

import pytest  # noqa: E402  (the env var above must come first)

from app.github_client import GitHubError, ReviewThread  # noqa: E402
from app.schemas import PullRequest  # noqa: E402
from fakes import FakeChatModel, Responder  # noqa: E402

LLM_SEAMS = ["app.graph.nodes.get_model", "app.graph.agent.get_model", "app.graph.verifier.get_model"]
GITHUB_SEAMS = [
    "app.graph.nodes.get_pull_request",
    "app.graph.nodes.get_posted_fingerprints",
    "app.graph.nodes.get_open_bot_threads",
    "app.graph.nodes.resolve_thread",
    "app.graph.nodes.get_file_text",
    "app.graph.tools.get_file_text",
    "app.api.main.get_pr_head",
    "app.graph.nodes.installation_token",
    "app.graph.nodes.post_review",
]


@pytest.fixture(autouse=True)
def no_real_apis(monkeypatch):
    """Safety net for every test: reaching OpenAI or GitHub fails loudly.

    Tests that need a model or a PR install fakes over this (fake_llm, fake_github).
    """

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to call a real API; install fake_llm / fake_github")

    for seam in LLM_SEAMS + GITHUB_SEAMS:
        monkeypatch.setattr(seam, refuse)


@pytest.fixture
def fake_llm(monkeypatch):
    """Usage: llm = fake_llm(scripted(...)); afterwards inspect llm.calls."""

    def install(respond: Responder) -> FakeChatModel:
        model = FakeChatModel(respond=respond)
        for seam in LLM_SEAMS:
            monkeypatch.setattr(seam, lambda role="smart", **kwargs: model)
        return model

    return install


@pytest.fixture
def fake_github(monkeypatch):
    """Usage: resolved = fake_github(pr, files={"path": "text"}, posted={"fp"}, threads=[...]).

    Unknown paths act like a 404. `posted`: fingerprints our bot already
    commented on the PR; `threads`: its open threads (none by default).
    Returns the ids of the threads that got resolved.
    """

    def install(pr: PullRequest, files: dict[str, str] | None = None, posted: set[str] = frozenset(),
                threads: list[ReviewThread] = ()) -> list[str]:
        files = files or {}
        resolved: list[str] = []

        def get_file_text(owner, repo, sha, path):
            if path not in files:
                raise GitHubError(f"{path} not found at {sha}")
            return files[path]

        monkeypatch.setattr("app.graph.nodes.get_pull_request", lambda url: pr)
        monkeypatch.setattr("app.graph.nodes.get_posted_fingerprints", lambda pr: set(posted))
        monkeypatch.setattr("app.graph.tools.get_file_text", get_file_text)
        monkeypatch.setattr("app.graph.nodes.get_file_text", get_file_text)
        monkeypatch.setattr("app.graph.nodes.get_open_bot_threads", lambda pr, token: list(threads))
        monkeypatch.setattr("app.graph.nodes.resolve_thread", lambda token, thread_id: resolved.append(thread_id))
        return resolved

    return install
