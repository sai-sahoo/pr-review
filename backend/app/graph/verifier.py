"""Verify findings before anyone sees them: the critic pattern.

Two layers, cheapest first:
1. grounding: plain Python. Is the file in the PR, is the line in the diff?
2. critic: one LLM call that scores every surviving finding.
The LLM gives evidence and a score; our code applies the threshold.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.diff import diff_line_numbers
from app.graph.prompts import VERIFIER_PROMPT
from app.llm import get_model
from app.schemas import Finding, FindingCheck, PullRequest, VerifierReport

MIN_CONFIDENCE = 0.6  # tune here, not in the prompt


def check_grounding(pr: PullRequest, findings: list[Finding]) -> tuple[list[Finding], list[FindingCheck]]:
    """Drop findings that point outside the diff. Returns (survivors, drops)."""
    visible = {f.path: diff_line_numbers(f.hunks) for f in pr.files}
    survivors: list[Finding] = []
    dropped: list[FindingCheck] = []
    for f in findings:
        if f.file not in visible:
            reason = f"{f.file} is not one of the reviewed files"
        elif f.line not in visible[f.file]:
            reason = f"line {f.line} is not in the diff of {f.file}"
        else:
            survivors.append(f)
            continue
        dropped.append(FindingCheck(finding=f, kept=False, stage="grounding", confidence=None, reason=reason))
    return survivors, dropped


def format_findings(findings: list[Finding]) -> str:
    """Number the findings so each verdict can point back by id."""
    blocks = [
        f"[{i}] {f.category}/{f.severity} - {f.file}:{f.line}\n"
        f"title: {f.title}\nexplanation: {f.explanation}\nsuggestion: {f.suggestion}"
        for i, f in enumerate(findings)
    ]
    return "\n\n".join(blocks)


def run_critic(findings: list[Finding], diff_text: str) -> list[FindingCheck]:
    """One structured call that scores every finding against the diff."""
    if not findings:  # nothing to check: don't pay for a call
        return []

    critic = get_model("smart").with_structured_output(VerifierReport)
    prompt = f"{diff_text}\n\n=== Findings to verify ===\n\n{format_findings(findings)}"
    report = critic.invoke([SystemMessage(VERIFIER_PROMPT), HumanMessage(prompt)])
    verdicts = {v.id: v for v in report.verdicts}  # ids we never asked about are ignored

    checks: list[FindingCheck] = []
    for i, f in enumerate(findings):
        v = verdicts.get(i)
        if v is None:
            # Fail open: missing a real bug costs more than showing a doubtful one.
            checks.append(FindingCheck(
                finding=f, kept=True, stage="verifier", confidence=None,
                reason="verifier returned no verdict; kept unverified",
            ))
            continue
        checks.append(FindingCheck(
            finding=f, kept=v.confidence >= MIN_CONFIDENCE, stage="verifier",
            confidence=v.confidence, reason=v.reason,
        ))
    return checks


def verify_findings(pr: PullRequest, findings: list[Finding], diff_text: str) -> list[FindingCheck]:
    """Run both layers. Returns one FindingCheck per input finding."""
    grounded, grounding_drops = check_grounding(pr, findings)
    return run_critic(grounded, diff_text) + grounding_drops
