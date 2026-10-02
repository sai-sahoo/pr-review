"""The verifier: grounding is plain Python; the critic uses a scripted fake model."""

from app.graph.verifier import MIN_CONFIDENCE, check_grounding, run_critic
from fakes import scripted, tool_call
from sample_pr import finding, make_pr


def verdict(id: int, confidence: float, reason: str = "checked") -> dict:
    return {"id": id, "reason": reason, "confidence": confidence}


def test_grounding_keeps_removed_check_and_drops_lines_outside_the_diff():
    removed_check = finding(line=11)  # cited at the nearest numbered line: valid
    outside = finding(line=99)
    other_file = finding(file="app/other.py", line=1)

    survivors, dropped = check_grounding(make_pr(), [removed_check, outside, other_file])

    assert survivors == [removed_check]
    assert [d.reason for d in dropped] == [
        "line 99 is not in the diff of app/payments.py",
        "app/other.py is not one of the reviewed files",
    ]
    assert all(d.stage == "grounding" and not d.kept for d in dropped)


def test_critic_threshold_is_applied_by_code(fake_llm):
    findings = [finding(line=11), finding(line=12), finding(line=13)]
    llm = fake_llm(scripted(tool_call("VerifierReport", verdicts=[
        verdict(0, 0.9),
        verdict(1, MIN_CONFIDENCE - 0.01),  # just below: dropped
        verdict(2, MIN_CONFIDENCE),  # exactly at the threshold: kept
    ])))

    checks = run_critic(findings, "<diff>")

    assert [c.kept for c in checks] == [True, False, True]
    assert [c.confidence for c in checks] == [0.9, MIN_CONFIDENCE - 0.01, MIN_CONFIDENCE]
    # The critic was shown the diff and every finding with its id.
    prompt = llm.calls[0].messages[1].content
    assert "<diff>" in prompt and "[0]" in prompt and "[2]" in prompt


def test_missing_verdict_fails_open_and_unknown_ids_are_ignored(fake_llm):
    findings = [finding(line=11), finding(line=12)]
    fake_llm(scripted(tool_call("VerifierReport", verdicts=[
        verdict(0, 0.1),
        verdict(7, 0.9),  # no finding [7]: ignored
    ])))

    first, second = run_critic(findings, "<diff>")

    assert not first.kept
    assert second.kept and second.confidence is None  # kept, but marked unscored


def test_no_findings_means_no_llm_call():
    # No fake installed: the safety net in conftest would fail any model call.
    assert run_critic([], "<diff>") == []
