"""The state that flows through the graph.

Every node receives the state and returns a dict with only the keys it
changed. LangGraph merges that dict into the state before the next step.
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from app.schemas import Finding, FindingCheck, PullRequest, Specialist, TriagePlan


class ReviewState(TypedDict):
    pr_url: str  # input: the only key the caller provides
    pr: NotRequired[PullRequest]  # written by fetch_pr
    plan: NotRequired[TriagePlan]  # written by triage

    # Several specialists write this key in the same step. The reducer says how
    # to combine their updates: operator.add(old, new) == old + new, i.e. append.
    # Without it, LangGraph raises InvalidUpdateError on parallel writes.
    raw_findings: Annotated[list[Finding], operator.add]

    tool_log: Annotated[list[str], operator.add]  # each specialist appends what it did

    findings: NotRequired[list[Finding]]  # written by aggregate: deduped and sorted

    # written by verify
    checks: NotRequired[list[FindingCheck]]  # every finding with its verdict, kept or not
    verified: NotRequired[list[Finding]]  # only the kept ones, sorted: the final answer


class SpecialistInput(TypedDict):
    """The private input one Send gives one specialist run (not the shared state)."""

    pr: PullRequest
    focus: Specialist
