"""The whole graph end to end, with fake GitHub and a fake model per agent."""

from app.graph import build_graph
from app.graph.prompts import SPECIALIST_PROMPTS, TRIAGE_PROMPT, VERIFIER_PROMPT
from fakes import by_system_prompt, scripted, tool_call
from sample_pr import finding, make_pr


def plan(*specialists: str):
    return scripted(tool_call("TriagePlan", reason="test plan", specialists=list(specialists)))


def submit(*findings):
    return scripted(tool_call("Review", findings=[f.model_dump() for f in findings]))


def run(pr_url: str = "https://github.com/acme/shop/pull/7") -> dict:
    return build_graph().invoke({"pr_url": pr_url})


def test_full_review(fake_llm, fake_github):
    fake_github(make_pr())
    removed_check_sec = finding(line=11, severity="high", category="security")
    removed_check_bug = finding(line=11, severity="critical")  # same line: dedupe
    hallucinated = finding(line=99, severity="medium", title="Line not in diff")
    weak = finding(line=12, severity="low", title="Maybe balance goes negative")

    llm = fake_llm(by_system_prompt({
        TRIAGE_PROMPT: plan("security", "correctness"),
        SPECIALIST_PROMPTS["security"]: submit(removed_check_sec),
        SPECIALIST_PROMPTS["correctness"]: submit(removed_check_bug, hallucinated, weak),
        # The critic sees what survived dedupe + grounding: [0] critical:11, [1] low:12
        VERIFIER_PROMPT: scripted(tool_call("VerifierReport", verdicts=[
            {"id": 0, "reason": "check really removed", "confidence": 0.95},
            {"id": 1, "reason": "speculative", "confidence": 0.3},
        ])),
    }))

    state = run()

    assert len(state["raw_findings"]) == 4  # both specialists' lists, appended by the reducer
    assert state["findings"] == [removed_check_bug, hallucinated, weak]  # deduped, sorted
    assert state["verified"] == [removed_check_bug]
    assert {(c.finding.line, c.stage, c.kept) for c in state["checks"]} == {
        (11, "verifier", True),
        (12, "verifier", False),
        (99, "grounding", False),
    }
    # Routing: only the planned specialists ran (maintainability's prompt was never sent).
    systems = {c.messages[0].content for c in llm.calls}
    assert SPECIALIST_PROMPTS["maintainability"] not in systems
    assert len(llm.calls) == 4  # triage + 2 specialists + critic


def test_empty_plan_skips_specialists_and_critic(fake_llm, fake_github):
    fake_github(make_pr())
    llm = fake_llm(by_system_prompt({TRIAGE_PROMPT: plan()}))

    state = run()

    assert state["verified"] == [] and state["tool_log"] == []
    assert len(llm.calls) == 1  # only triage


def test_pr_with_nothing_to_review_makes_no_llm_calls(fake_github):
    pr = make_pr()
    fake_github(pr.model_copy(update={"files": []}))
    # No fake_llm: the conftest safety net fails the test on any model call.

    state = run()

    assert state["plan"].specialists == []
    assert state["verified"] == []
