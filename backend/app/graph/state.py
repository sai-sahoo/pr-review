"""The state that flows through the graph.

Every node receives the state and returns a dict with only the keys it
changed. LangGraph merges that dict into the state before the next step.
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from app.schemas import Finding, FindingCheck, PullRequest, Specialist, TriagePlan


class ReviewState(TypedDict):
    pr_url: str  # input
    # input, optional: fingerprints of findings a human dismissed in earlier
    # reviews of this PR. The worker reads them from Postgres.
    known_dismissed: NotRequired[list[str]]
    pr: NotRequired[PullRequest]  # written by fetch_pr
    plan: NotRequired[TriagePlan]  # written by triage

    # Several specialists write this key in the same step. The reducer says how
    # to combine their updates: operator.add(old, new) == old + new, i.e. append.
    # Without it, LangGraph raises InvalidUpdateError on parallel writes.
    raw_findings: Annotated[list[Finding], operator.add]

    tool_log: Annotated[list[str], operator.add]  # each specialist appends what it did

    findings: NotRequired[list[Finding]]  # written by aggregate: deduped and sorted

    # written by verify, then narrowed by skip_seen to what's new on this PR
    checks: NotRequired[list[FindingCheck]]  # every finding with its verdict, kept or not
    verified: NotRequired[list[Finding]]  # only the kept ones, sorted: the final answer

    # written by approve: the verified findings a human chose to post. Missing
    # when nobody was asked (nothing to post, or nowhere to post it).
    approved: NotRequired[list[Finding]]
    dismissed: NotRequired[list[str]]  # written by approve: fingerprints of the rest

    # written by publish
    review_url: NotRequired[str | None]  # the review on GitHub, or None if not posted
    publish_note: NotRequired[str]  # what happened, in words: posted, or why not

    # written by resolve_fixed
    resolved: NotRequired[list[str]]  # the file of each thread marked resolved
    resolve_note: NotRequired[str]  # what happened, in words


class SpecialistInput(TypedDict):
    """The private input one Send gives one specialist run (not the shared state)."""

    pr: PullRequest
    focus: Specialist
