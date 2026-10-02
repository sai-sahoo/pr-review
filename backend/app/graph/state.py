"""The state that flows through the graph.

Every node receives the whole state and returns a dict with only the keys
it changed. LangGraph merges that dict into the state before the next node.
"""

from typing import NotRequired, TypedDict

from app.schemas import Finding, PullRequest


class ReviewState(TypedDict):
    pr_url: str  # input: the only key the caller provides
    pr: NotRequired[PullRequest]  # written by fetch_pr
    findings: NotRequired[list[Finding]]  # written by review
