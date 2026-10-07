"""Step 15b-1: a re-review skips what's already on the PR or was dismissed before."""

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.fingerprint import MARKER_RE, fingerprint
from app.github_client import GitHubError, get_posted_fingerprints, review_payload
from app.graph import build_graph
from app.graph.prompts import SPECIALIST_PROMPTS, TRIAGE_PROMPT, VERIFIER_PROMPT
from fakes import by_system_prompt, scripted, tool_call
from sample_pr import PATH, REMOVED_CHECK_DIFF, finding, make_pr

PR_URL = "https://github.com/acme/shop/pull/7"
CONFIG = {"configurable": {"thread_id": "t1"}}

# --- the fingerprint -------------------------------------------------------


def test_fingerprint_survives_shifted_lines():
    # A later push added 5 lines above the function: the same code is now at
    # line 16, with different indentation width, and the LLM worded it differently.
    shifted = make_pr(REMOVED_CHECK_DIFF.replace("@@ -10,6 +10,5 @@", "@@ -10,6 +15,5 @@")
                      .replace("+    log.info(", "+        log.info("))  # re-indented
    before = finding(line=11)
    after = finding(line=16, title="Missing validation", category="security")

    assert fingerprint(make_pr(), before) == fingerprint(shifted, after)


def test_different_code_is_a_different_finding():
    pr = make_pr()
    assert fingerprint(pr, finding(line=11)) != fingerprint(pr, finding(line=12))


def test_posted_comments_carry_the_fingerprint():
    pr = make_pr()
    [comment] = review_payload(pr, [finding(line=11)])["comments"]

    assert MARKER_RE.findall(comment["body"]) == [fingerprint(pr, finding(line=11))]


# --- reading them back from GitHub -----------------------------------------


def comment(body: str, user_type: str = "Bot") -> dict:
    return {"body": body, "user": {"login": "x", "type": user_type}}


def test_reads_markers_from_bot_comments_on_every_page(monkeypatch):
    seen_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.params.get("page"))
        if request.url.params.get("page") != "2":
            return httpx.Response(200, json=[
                comment("🔴 bug\n\n<!-- pr-review:aaaaaaaaaaaa -->"),
                comment("> quoting it: <!-- pr-review:bbbbbbbbbbbb -->", user_type="User"),  # a human
            ], headers={"Link": '<https://api.github.com/repos/acme/shop/pulls/7/comments?per_page=100&page=2>; rel="next"'})
        return httpx.Response(200, json=[comment("<!-- pr-review:cccccccccccc -->"), comment("no marker")])

    monkeypatch.setattr("app.github_client._client", lambda owner, repo: httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)))

    assert get_posted_fingerprints(make_pr()) == {"aaaaaaaaaaaa", "cccccccccccc"}
    assert seen_paths == [None, "2"]  # followed the Link header to the last page


# --- in the graph ----------------------------------------------------------


@pytest.fixture
def installed_app(monkeypatch):
    """The App can post here. Returns what got posted."""
    monkeypatch.setenv("GITHUB_APP_ID", "1")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "unused.pem")
    monkeypatch.setattr("app.graph.nodes.installation_token", lambda owner, repo: "ghs_abc")
    posted: list[dict] = []
    monkeypatch.setattr("app.graph.nodes.post_review",
                        lambda pr, payload, token: posted.append(payload) or "https://github.com/r/1")
    return posted


def script_two_findings(fake_llm):
    """The correctness specialist reports lines 11 and 12; the critic keeps both."""
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


def fp(line: int) -> str:
    return fingerprint(make_pr(), finding(line=line))


def test_already_posted_finding_is_not_asked_about_or_posted_again(fake_llm, fake_github, installed_app):
    script_two_findings(fake_llm)
    fake_github(make_pr(), posted={fp(11)})  # an earlier review commented on line 11
    graph = build_graph(InMemorySaver())

    graph.invoke({"pr_url": PR_URL}, CONFIG)

    # Only the new one is up for approval: position 0 is now line 12.
    [pause] = graph.get_state(CONFIG).interrupts
    assert [f["line"] for f in pause.value["findings"]] == [12]
    state = graph.invoke(Command(resume={"approved": [0]}), CONFIG)

    [payload] = installed_app
    assert [c["line"] for c in payload["comments"]] == [12]
    seen = [c for c in state["checks"] if c.stage == "seen"]
    assert [(c.finding.line, c.kept, c.reason) for c in seen] == [(11, False, "already commented on this PR")]


def test_finding_dismissed_before_is_skipped(fake_llm, fake_github, installed_app):
    script_two_findings(fake_llm)
    fake_github(make_pr())
    graph = build_graph(InMemorySaver())

    graph.invoke({"pr_url": PR_URL, "known_dismissed": [fp(12)]}, CONFIG)

    [pause] = graph.get_state(CONFIG).interrupts
    assert [f["line"] for f in pause.value["findings"]] == [11]


def test_nothing_new_posts_nothing(fake_llm, fake_github, installed_app):
    script_two_findings(fake_llm)
    fake_github(make_pr(), posted={fp(11), fp(12)})
    graph = build_graph(InMemorySaver())

    state = graph.invoke({"pr_url": PR_URL}, CONFIG)

    assert graph.get_state(CONFIG).next == ()  # no pause: nothing to ask about
    assert installed_app == []  # and no "no issues ✅": the earlier comments are still open
    assert state["publish_note"] == "not posted: nothing new since the earlier reviews"


def test_approval_remembers_what_was_dismissed(fake_llm, fake_github, installed_app):
    script_two_findings(fake_llm)
    fake_github(make_pr())
    graph = build_graph(InMemorySaver())
    graph.invoke({"pr_url": PR_URL}, CONFIG)

    state = graph.invoke(Command(resume={"approved": [0]}), CONFIG)

    assert state["dismissed"] == [fp(12)]  # the worker stores this on the review


def test_unreadable_comments_mean_everything_counts_as_new(fake_llm, fake_github, monkeypatch):
    script_two_findings(fake_llm)
    fake_github(make_pr())

    def refused(pr):
        raise GitHubError("GitHub refused the request (likely rate limit).")

    monkeypatch.setattr("app.graph.nodes.get_posted_fingerprints", refused)

    state = build_graph().invoke({"pr_url": PR_URL})  # no App: runs straight through

    assert [f.line for f in state["verified"]] == [11, 12]  # the review still finishes
