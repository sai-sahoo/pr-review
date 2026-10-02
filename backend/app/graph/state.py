"""The state that flows through the graph.

Every node receives the state and returns a dict with only the keys it
changed. LangGraph merges that dict into the state before the next step.
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from app.schemas import Finding, PullRequest, Specialist, TriagePlan


class ReviewState(TypedDict):
    pr_url: str  # input: the only key the caller provides
    pr: NotRequired[PullRequest]  # written by fetch_pr
    plan: NotRequired[TriagePlan]  # written by triage

    # Several specialists write this key in the same step. The reducer says how
    # to combine their updates: operator.add(old, new) == old + new, i.e. append.
    # Without it, LangGraph raises InvalidUpdateError on parallel writes.
    raw_findings: Annotated[list[Finding], operator.add]

    findings: NotRequired[list[Finding]]  # written by aggregate: deduped and sorted


class SpecialistInput(TypedDict):
    """The private input one Send gives one specialist run (not the shared state)."""

    pr: PullRequest
    focus: Specialist
