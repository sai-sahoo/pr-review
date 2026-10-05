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

import pytest  # noqa: E402  (the env var above must come first)

from app.github_client import GitHubError  # noqa: E402
from app.schemas import PullRequest  # noqa: E402
from fakes import FakeChatModel, Responder  # noqa: E402

LLM_SEAMS = ["app.graph.nodes.get_model", "app.graph.agent.get_model", "app.graph.verifier.get_model"]
GITHUB_SEAMS = ["app.graph.nodes.get_pull_request", "app.graph.tools.get_file_text"]


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
    """Usage: fake_github(pr, files={"path": "text"}). Unknown paths act like a 404."""

    def install(pr: PullRequest, files: dict[str, str] | None = None) -> None:
        files = files or {}

        def get_file_text(owner, repo, sha, path):
            if path not in files:
                raise GitHubError(f"{path} not found at {sha}")
            return files[path]

        monkeypatch.setattr("app.graph.nodes.get_pull_request", lambda url: pr)
        monkeypatch.setattr("app.graph.tools.get_file_text", get_file_text)

    return install
