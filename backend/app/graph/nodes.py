"""Graph nodes: plain functions, state in -> partial update out.

Nodes don't know about each other or about edges. The wiring lives in
build.py, which is why the same `specialist` node can run three times at once.
"""

import logging

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.types import Send, interrupt

from app.diff import number_hunk
from app import github_app
from app.fingerprint import code_fingerprint, fingerprint
from app.github_app import installation_token
from app.github_client import (
    GitHubError,
    ReviewThread,
    get_file_text,
    get_open_bot_threads,
    get_posted_fingerprints,
    get_pull_request,
    post_review,
    resolve_thread,
    review_payload,
)
from app.graph.agent import run_specialist
from app.graph.prompts import TRIAGE_PROMPT
from app.graph.state import ReviewState, SpecialistInput
from app.graph.verifier import verify_findings
from app.llm import get_model
from app.schemas import SEVERITY_RANK, Finding, FindingCheck, PullRequest, TriagePlan

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


def skip_seen(state: ReviewState) -> dict:
    """Re-reviews like a human: don't raise again what was already raised.

    Every push to a PR gets a full new review, and most of its findings are
    the ones from last time. A finding is dropped here if
      - our bot already commented it on this PR (read back from GitHub), or
      - a human dismissed it at the approval step of an earlier review.
    Matching is by fingerprint (app/fingerprint.py), since line numbers
    shift between pushes.

    It narrows `verified` and marks the dropped ones in `checks`, so
    everything after it (approve, publish, the UI, the API) works unchanged.
    Positions in the approval step count only what's new.
    """
    pr = state["pr"]
    dismissed = set(state.get("known_dismissed", []))
    try:
        posted = get_posted_fingerprints(pr)
    except (GitHubError, httpx.HTTPError) as e:
        # Not worth failing a finished review over. The cost is some repeated
        # comments, which the human can untick at the approval step.
        log.warning("could not read earlier comments on %s, treating every finding as new: %s", pr.url, e)
        posted = set()

    checks: list[FindingCheck] = []
    for c in state["checks"]:
        if c.kept:
            fp = fingerprint(pr, c.finding)
            if fp in posted:
                c = c.model_copy(update={"kept": False, "stage": "seen", "reason": "already commented on this PR"})
            elif fp in dismissed:
                c = c.model_copy(update={"kept": False, "stage": "seen",
                                         "reason": "dismissed in an earlier review of this PR"})
        checks.append(c)
    # Same order as verify's list, so the severity order still holds.
    return {"checks": checks, "verified": [c.finding for c in checks if c.kept]}


def can_post(pr: PullRequest) -> bool:
    """Would publish be able to post to this PR? Only then is a human asked."""
    if not github_app.configured():
        return False
    try:
        return installation_token(pr.owner, pr.repo) is not None
    except (GitHubError, httpx.HTTPError, OSError):
        return False  # don't make anyone approve a post that can't happen; publish says why


def route_to_approval(state: ReviewState) -> str:
    """Conditional edge: ask a human only when there's something to post and
    somewhere to post it. Otherwise straight on to publish, which explains.

    Deciding here, not inside approve, matters on resume: a router's choice
    is saved in skip_seen's checkpoint and never re-made. approve itself runs
    again from its first line when resumed (see below), so it must not
    contain a check whose answer could change while it waits, like "is the
    App still installed?".
    """
    if state["verified"] and can_post(state["pr"]):
        return "approve"
    return "publish"


def approve(state: ReviewState) -> dict:
    """Human in the loop: pause until someone decides which findings get posted.

    The first time through, interrupt() doesn't return. It saves the run
    (the checkpointer already holds the state) and stops the graph; the
    value passed to it travels out to whoever is running the graph.

    Later, someone runs the graph again with Command(resume=decision). The
    node then starts *over from its first line*, and this time interrupt()
    returns the decision instead of stopping. So anything above the
    interrupt() runs twice: keep it cheap and give it the same answer both
    times.
    """
    verified = state["verified"]
    decision = interrupt({"findings": [f.model_dump(mode="json") for f in verified]})
    # decision is {"approved": [0, 2]}: positions in `verified`. The API has
    # already checked them; a set makes "is i approved?" a quick lookup.
    keep = set(decision["approved"])
    return {
        "approved": [f for i, f in enumerate(verified) if i in keep],
        # Remembered (the worker stores them on the review), so the next push
        # doesn't ask about these again: skip_seen gets them as known_dismissed.
        "dismissed": [fingerprint(state["pr"], f) for i, f in enumerate(verified) if i not in keep],
    }


def publish(state: ReviewState) -> dict:
    """Post the verified findings to the PR as one review, as the GitHub App.

    A node rather than a step after the graph, for the checkpointer: once
    publish has finished, a resumed run never reaches it again, so a worker
    restart can't post the same review twice.

    Posts what approve let through. A human has already said yes to each
    one when the run paused there.

    Never fails the review: the findings are already good, and they're saved
    either way. Whatever happens ends up in publish_note.
    """
    pr = state["pr"]
    # What a human approved, or, when nobody was asked, everything verified.
    findings = state.get("approved", state["verified"])
    if state["verified"] and not findings:
        # Not the "no issues ✅" review: that would claim the PR is clean.
        return {"review_url": None, "publish_note": "not posted: every finding was dismissed"}
    if not findings and any(c.stage == "seen" for c in state.get("checks", [])):
        # Same reason: the earlier comments are still open, so the PR isn't clean.
        # And a "nothing new" review on every push would be noise.
        return {"review_url": None, "publish_note": "not posted: nothing new since the earlier reviews"}
    if not github_app.configured():
        return {"review_url": None, "publish_note": "not posted: no GitHub App configured"}
    try:
        token = installation_token(pr.owner, pr.repo)
        if token is None:
            return {"review_url": None, "publish_note": f"not posted: the App isn't installed on {pr.owner}/{pr.repo}"}
        url = post_review(pr, review_payload(pr, findings), token)
    except (GitHubError, httpx.HTTPError, OSError) as e:  # OSError: e.g. the .pem file is missing
        log.exception("could not post the review of %s", pr.url)
        return {"review_url": None, "publish_note": f"could not post: {e}"}
    return {"review_url": url, "publish_note": "posted"}


def code_is_gone(pr: PullRequest, thread: ReviewThread) -> bool:
    """Is the line this thread is about no longer anywhere in its file?

    Compared by fingerprint against every line of the file at the reviewed
    commit, so a line that only moved (code added above it) still counts as
    there. Any doubt means "still there": a wrongly open thread costs a
    click, a wrongly resolved one hides a real bug.
    """
    if any(s.path == thread.path and s.reason == "deleted" for s in pr.skipped):
        return True  # the PR deletes the whole file
    try:
        text = get_file_text(pr.owner, pr.repo, pr.head_sha, thread.path)
    except (GitHubError, httpx.HTTPError):
        return False  # can't tell (rate limit, or a rename): leave it open
    return thread.fingerprint not in {code_fingerprint(thread.path, line) for line in text.splitlines()}


def resolve_fixed(state: ReviewState) -> dict:
    """Re-reviews like a human, part 2: resolve our own threads that are fixed.

    A thread counts as fixed when the line it points at is gone from the
    file. That means it was edited or deleted. Two things this deliberately
    doesn't use:
      - "this run didn't report it again": the LLM misses things from run to
        run, so that alone would resolve real bugs. (While the line is
        there, the same finding has the same fingerprint, so "gone" already
        implies "can't be reported again".)
      - GitHub's own "Outdated" label: it appears whenever the diff around
        a comment changes, even when the commented line itself didn't.

    Runs after publish, without asking: it only tidies the bot's own
    comments, keeps them on the PR, and anyone can unresolve one in a click.
    Like publish it never fails the review, and running it twice is harmless.
    """
    pr = state["pr"]
    if not github_app.configured():
        return {"resolved": [], "resolve_note": "skipped: no GitHub App configured"}
    try:
        token = installation_token(pr.owner, pr.repo)
        if token is None:
            return {"resolved": [], "resolve_note": f"skipped: the App isn't installed on {pr.owner}/{pr.repo}"}
        fixed = [t for t in get_open_bot_threads(pr, token) if code_is_gone(pr, t)]
        resolved: list[str] = []
        for t in fixed:
            resolve_thread(token, t.id)
            resolved.append(t.path)
    except (GitHubError, httpx.HTTPError, OSError) as e:
        log.exception("could not resolve threads on %s", pr.url)
        return {"resolved": [], "resolve_note": f"could not resolve threads: {e}"}
    return {"resolved": resolved, "resolve_note": f"resolved {len(resolved)} fixed thread{'s' * (len(resolved) != 1)}"}
