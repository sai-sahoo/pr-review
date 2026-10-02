"""Graph nodes: plain functions, state in -> partial update out.

Nodes don't know about each other or about edges. That's why they are easy
to test, reorder, or run in parallel later (Step 5).
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.diff import number_hunk
from app.github_client import get_pull_request
from app.graph.state import ReviewState
from app.llm import get_model
from app.schemas import PullRequest, Review

SYSTEM_PROMPT = """\
You are a senior code reviewer. Review the pull request diff you are given.
- Report only real problems introduced by the added (+) lines.
- Each line starts with its line number in the new file; use that number for `line`.
- No style nitpicks and no invented issues. An empty list is a valid answer."""

MAX_DESCRIPTION_CHARS = 2000  # PR bodies can be huge templates; the diff matters more


def format_pr(pr: PullRequest) -> str:
    """Render the PR as the text the LLM will read."""
    parts = [f"PR #{pr.number}: {pr.title}"]
    if pr.description:
        parts.append(f"Description:\n{pr.description[:MAX_DESCRIPTION_CHARS]}")
    for f in pr.files:
        parts.append(f"=== {f.path} ({f.status}) ===")
        parts.extend(number_hunk(h) for h in f.hunks)
    return "\n\n".join(parts)


def fetch_pr(state: ReviewState) -> dict:
    return {"pr": get_pull_request(state["pr_url"])}


def review(state: ReviewState) -> dict:
    pr = state["pr"]
    if not pr.files:  # everything was filtered out: nothing to pay an LLM for
        return {"findings": []}

    reviewer = get_model("smart").with_structured_output(Review)
    result = reviewer.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(format_pr(pr))])
    return {"findings": result.findings}
