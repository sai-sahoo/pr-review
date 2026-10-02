"""Graph nodes: plain functions, state in -> partial update out.

Nodes don't know about each other or about edges. The wiring lives in
build.py, which is why the same `specialist` node can run three times at once.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from app.diff import number_hunk
from app.github_client import get_pull_request
from app.graph.prompts import SPECIALIST_PROMPTS, TRIAGE_PROMPT
from app.graph.state import ReviewState, SpecialistInput
from app.llm import get_model
from app.schemas import SEVERITY_RANK, Finding, PullRequest, Review, TriagePlan

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
    reviewer = get_model("smart").with_structured_output(Review)
    result = reviewer.invoke(
        [SystemMessage(SPECIALIST_PROMPTS[state["focus"]]), HumanMessage(format_pr(state["pr"]))]
    )
    # Goes through the operator.add reducer: appended, not overwritten.
    return {"raw_findings": result.findings}


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
