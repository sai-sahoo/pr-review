"""Step 15: the run pauses at approve, and only what a human approves is posted.

These drive the graph directly, the way the CLI does: run until the pause,
look at the checkpoint, resume with Command(resume=...). test_api.py covers
the same flow through the worker and POST /reviews/{id}/approval.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import build_graph
from app.graph.prompts import SPECIALIST_PROMPTS, TRIAGE_PROMPT, VERIFIER_PROMPT
from fakes import by_system_prompt, scripted, tool_call
from sample_pr import finding, make_pr

PR_URL = "https://github.com/acme/shop/pull/7"
CONFIG = {"configurable": {"thread_id": "t1"}}


@pytest.fixture
def installed_app(monkeypatch):
    """The App is configured and installed on the PR's repo. Returns what got posted."""
    monkeypatch.setenv("GITHUB_APP_ID", "1")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "unused.pem")  # installation_token is faked
    monkeypatch.setattr("app.graph.nodes.installation_token", lambda owner, repo: "ghs_abc")
    posted: list[dict] = []
    monkeypatch.setattr("app.graph.nodes.post_review",
                        lambda pr, payload, token: posted.append(payload) or "https://github.com/r/1")
    return posted


def two_findings(fake_llm, fake_github):
    """One specialist reports lines 11 and 12; the critic keeps both."""
    fake_github(make_pr())
    fake_llm(by_system_prompt({
        TRIAGE_PROMPT: scripted(tool_call("TriagePlan", reason="r", specialists=["correctness"])),
        SPECIALIST_PROMPTS["correctness"]: scripted(tool_call("Review", findings=[
            finding(line=11, severity="critical").model_dump(),
            finding(line=12, severity="low", title="Balance can go negative").model_dump(),
        ])),
        VERIFIER_PROMPT: scripted(tool_call("VerifierReport", verdicts=[
            {"id": 0, "reason": "real", "confidence": 0.9},
            {"id": 1, "reason": "real", "confidence": 0.9},
        ])),
    }))


def test_pauses_before_posting_then_posts_only_what_was_approved(fake_llm, fake_github, installed_app):
    two_findings(fake_llm, fake_github)
    graph = build_graph(InMemorySaver())

    graph.invoke({"pr_url": PR_URL}, CONFIG)

    # Stopped inside approve, with the findings handed out; nothing posted yet.
    snapshot = graph.get_state(CONFIG)
    assert snapshot.next == ("approve",)
    [pause] = snapshot.interrupts
    assert [f["line"] for f in pause.value["findings"]] == [11, 12]
    assert installed_app == []

    # The human keeps the second one only. The scripted agents have no answers
    # left, so this would fail if anything before approve ran again.
    state = graph.invoke(Command(resume={"approved": [1]}), CONFIG)

    assert [f.line for f in state["approved"]] == [12]
    [payload] = installed_app
    assert [c["line"] for c in payload["comments"]] == [12]
    assert state["publish_note"] == "posted"
    assert graph.get_state(CONFIG).next == ()  # finished


def test_dismissing_everything_posts_nothing(fake_llm, fake_github, installed_app):
    two_findings(fake_llm, fake_github)
    graph = build_graph(InMemorySaver())
    graph.invoke({"pr_url": PR_URL}, CONFIG)

    state = graph.invoke(Command(resume={"approved": []}), CONFIG)

    assert installed_app == []  # and certainly not a "no issues ✅" review
    assert state["publish_note"] == "not posted: every finding was dismissed"


def test_running_again_without_a_decision_just_pauses_again(fake_llm, fake_github, installed_app):
    # What the worker does when it stopped right after the pause, before
    # marking the review "waiting": run with None. approve starts over and
    # interrupt() stops it again, since there's still no resume value.
    two_findings(fake_llm, fake_github)
    graph = build_graph(InMemorySaver())
    graph.invoke({"pr_url": PR_URL}, CONFIG)

    graph.invoke(None, CONFIG)

    assert graph.get_state(CONFIG).next == ("approve",)
    assert installed_app == []


def test_no_pause_where_the_app_cannot_post(fake_llm, fake_github, monkeypatch):
    two_findings(fake_llm, fake_github)
    monkeypatch.setenv("GITHUB_APP_ID", "1")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "unused.pem")
    monkeypatch.setattr("app.graph.nodes.installation_token", lambda owner, repo: None)
    graph = build_graph(InMemorySaver())

    state = graph.invoke({"pr_url": PR_URL}, CONFIG)

    assert graph.get_state(CONFIG).next == ()  # straight through: nobody to ask
    assert "approved" not in state
    assert state["publish_note"] == "not posted: the App isn't installed on acme/shop"


def test_no_pause_when_there_is_nothing_to_approve(fake_llm, fake_github, installed_app):
    fake_github(make_pr())
    fake_llm(by_system_prompt({TRIAGE_PROMPT: scripted(tool_call("TriagePlan", reason="r", specialists=[]))}))
    graph = build_graph(InMemorySaver())

    graph.invoke({"pr_url": PR_URL}, CONFIG)

    assert graph.get_state(CONFIG).next == ()
    [payload] = installed_app  # the "no issues ✅" review, posted without asking
    assert payload["comments"] == []
