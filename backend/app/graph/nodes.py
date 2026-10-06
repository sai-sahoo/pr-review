"""Graph nodes: plain functions, state in -> partial update out.

Nodes don't know about each other or about edges. The wiring lives in
build.py, which is why the same `specialist` node can run three times at once.
"""

import logging

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.types import Send

from app.diff import number_hunk
from app import github_app
from app.github_app import installation_token
from app.github_client import GitHubError, get_pull_request, post_review, review_payload
from app.graph.agent import run_specialist
from app.graph.prompts import TRIAGE_PROMPT
from app.graph.state import ReviewState, SpecialistInput
from app.graph.verifier import verify_findings
from app.llm import get_model
from app.schemas import SEVERITY_RANK, Finding, PullRequest, TriagePlan

log = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 2000  # PR bodies can be huge templates; the diff matters more


def format_pr(pr: PullRequest) -> str:
    """Render the full PR as the text a specialist reads."""
    parts = [f"PR #{pr.number}: {pr.title}"]
    if pr.description:
        parts.append(f"Description:\n{pr.description[:MAX_DESCRIPTION_CHARS]}")
    for f in pr.files:
        parts.append(f"=== {f.path} ({f.status}) ===")
        parts.extend(number_hunk(h) for h in f.hunks)
    return "\n\n".join(parts)


def format_file_list(pr: PullRequest) -> str:
    """Just titles and paths: enough to plan, and far cheaper than the diff."""
    lines = [f"PR #{pr.number}: {pr.title}", "", "Changed files:"]
    lines += [f"- {f.path} ({f.status}, +{f.added} -{f.removed})" for f in pr.files]
    return "\n".join(lines)


def fetch_pr(state: ReviewState) -> dict:
    return {"pr": get_pull_request(state["pr_url"])}


def triage(state: ReviewState) -> dict:
    pr = state["pr"]
    if not pr.files:  # everything was filtered out: nothing to pay an LLM for
        return {"plan": TriagePlan(reason="No reviewable files.", specialists=[])}

    planner = get_model("fast").with_structured_output(TriagePlan)
    plan = planner.invoke([SystemMessage(TRIAGE_PROMPT), HumanMessage(format_file_list(pr))])
    return {"plan": plan}


def route_to_specialists(state: ReviewState) -> list[Send] | str:
    """Conditional edge: one Send per chosen specialist, all run in parallel.

    Returning a node name instead of Sends is a normal jump, used here to
    skip straight to aggregate when the plan is empty.
    """
    chosen = dict.fromkeys(state["plan"].specialists)  # drop duplicates, keep order
    if not chosen:
        return "aggregate"
    return [Send("specialist", {"pr": state["pr"], "focus": name}) for name in chosen]


def specialist(state: SpecialistInput) -> dict:
    pr = state["pr"]
    # The writer sends custom events to stream_mode="custom" listeners while the
    # node is still running. With no such listener, writing is a no-op.
    findings, log = run_specialist(pr, state["focus"], format_pr(pr), emit=get_stream_writer())
    # Both keys go through operator.add reducers: appended, not overwritten.
    return {"raw_findings": findings, "tool_log": log}


def aggregate(state: ReviewState) -> dict:
    """Merge all specialists' findings: one per (file, line), the most severe wins.

    Writes a separate `findings` key. Writing back to `raw_findings` would go
    through the reducer and *append* the deduped list to the original.
    """
    best: dict[tuple[str, int], Finding] = {}
    for f in state["raw_findings"]:
        key = (f.file, f.line)
        if key not in best or SEVERITY_RANK[f.severity] < SEVERITY_RANK[best[key].severity]:
            best[key] = f
    findings = sorted(best.values(), key=lambda f: (SEVERITY_RANK[f.severity], f.file, f.line))
    return {"findings": findings}


def verify(state: ReviewState) -> dict:
    """Critic pass: drop findings that are ungrounded or not convincing."""
    pr = state["pr"]
    checks = verify_findings(pr, state["findings"], format_pr(pr))
    # Filtering keeps aggregate's severity order, so no re-sort needed.
    return {"checks": checks, "verified": [c.finding for c in checks if c.kept]}


def publish(state: ReviewState) -> dict:
    """Post the verified findings to the PR as one review, as the GitHub App.

    A node rather than a step after the graph, for the checkpointer: once
    publish has finished, a resumed run never reaches it again, so a worker
    restart can't post the same review twice.

    Never fails the review: the findings are already good, and they're saved
    either way. Whatever happens ends up in publish_note.
    """
    pr = state["pr"]
    if not github_app.configured():
        return {"review_url": None, "publish_note": "not posted: no GitHub App configured"}
    try:
        token = installation_token(pr.owner, pr.repo)
        if token is None:
            return {"review_url": None, "publish_note": f"not posted: the App isn't installed on {pr.owner}/{pr.repo}"}
        url = post_review(pr, review_payload(pr, state["verified"]), token)
    except (GitHubError, httpx.HTTPError, OSError) as e:  # OSError: e.g. the .pem file is missing
        log.exception("could not post the review of %s", pr.url)
        return {"review_url": None, "publish_note": f"could not post: {e}"}
    return {"review_url": url, "publish_note": "posted"}
