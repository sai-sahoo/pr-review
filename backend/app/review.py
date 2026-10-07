"""Review a real PR with a LangGraph:
    fetch_pr -> triage -> specialists in parallel -> aggregate -> verify -> skip_seen -> approve -> publish -> resolve_fixed
    (publish posts to the PR only if the GitHub App is configured and installed there,
     and then approve first asks you, right here in the terminal, which findings to post;
     resolve_fixed then resolves the bot's threads whose code is gone)

Run:  cd backend && uv run python -m app.review https://github.com/OWNER/REPO/pull/123
      add --show-graph to print the graph as a Mermaid diagram (no API calls)
"""

import argparse
import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.github_client import GitHubError
from app.graph import build_graph


def describe(node: str, update: dict) -> str:
    """One progress line for a node's state update."""
    if node == "fetch_pr":
        pr = update["pr"]
        return f"#{pr.number} {pr.title}: {len(pr.files)} files, {len(pr.skipped)} skipped"
    if node == "triage":
        plan = update["plan"]
        return f"run [{', '.join(plan.specialists)}]: {plan.reason}"
    if node == "specialist":
        # Last log line is the summary; the ones before it are the tool calls.
        *calls, summary = update["tool_log"]
        return "\n".join([summary, *(f"{'':19}-> {c}" for c in calls)])
    if node == "aggregate":
        return f"{len(update['findings'])} findings after dedupe"
    if node == "verify":
        checks = update["checks"]
        return f"kept {len(update['verified'])} of {len(checks)}"
    if node == "skip_seen":
        seen = sum(c.stage == "seen" for c in update["checks"])
        return f"{len(update['verified'])} new, {seen} already raised on this PR"
    if node == "approve":
        return f"posting {len(update['approved'])} you approved"
    if node == "publish":
        return update["review_url"] or update["publish_note"]
    if node == "resolve_fixed":
        return update["resolve_note"]
    return ""


def fmt_conf(confidence: float | None) -> str:
    return "unscored" if confidence is None else f"confidence {confidence:.2f}"


def ask_approval(findings: list[dict]) -> dict:
    """The CLI's approval screen: the interrupt's findings in, a decision out."""
    print("\n=== Post these findings to the PR? ===")
    for i, f in enumerate(findings, 1):  # numbered from 1 for people; positions are from 0
        print(f"  {i}. [{f['severity'].upper()}] {f['file']}:{f['line']}  {f['title']}")
    answer = input("Numbers to post (e.g. 1,3), 'all' or 'none' [all]: ").strip().lower()
    if answer in ("", "all"):
        return {"approved": list(range(len(findings)))}
    if answer == "none":
        return {"approved": []}
    picked = {int(n) - 1 for n in answer.replace(",", " ").split() if n.isdigit()}
    return {"approved": sorted(i for i in picked if 0 <= i < len(findings))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a GitHub PR with an LLM.")
    parser.add_argument("pr_url", nargs="?")
    parser.add_argument("--show-graph", action="store_true", help="print the graph and exit")
    args = parser.parse_args()

    # interrupt() needs a checkpointer to pause into. A run of this script
    # lives only as long as the script, so memory is enough; the worker uses Postgres.
    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "cli"}}

    if args.show_graph:
        print(graph.get_graph().draw_mermaid())  # paste into https://mermaid.live
        return
    if not args.pr_url:
        parser.error("pr_url is required")

    start = time.perf_counter()
    checks = []
    graph_input: dict | Command = {"pr_url": args.pr_url}
    try:
        # Each pass runs the graph until it ends or pauses. A pause at approve
        # ends the stream like a finish does; the checkpoint tells them apart.
        while True:
            # stream_mode="updates" yields {node_name: update} each time a node finishes,
            # so we can watch the graph run. invoke() would only return the end state.
            for chunk in graph.stream(graph_input, config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":  # the pause itself, not a node
                        continue
                    print(f"[{time.perf_counter() - start:5.1f}s] {node:<10} {describe(node, update)}")
                    if node in ("verify", "skip_seen"):
                        checks = update["checks"]
            interrupts = graph.get_state(config).interrupts
            if not interrupts:
                break
            # The value approve passed to interrupt(), and our answer back to it.
            graph_input = Command(resume=ask_approval(interrupts[0].value["findings"]))
    except GitHubError as e:  # node exceptions propagate out of the graph unchanged
        raise SystemExit(f"error: {e}")

    kept = [c for c in checks if c.kept]
    dropped = [c for c in checks if not c.kept]

    print(f"\n=== Findings ({len(kept)}) ===")
    for c in kept:  # aggregate already sorted them by severity
        f = c.finding
        print(f"\n[{f.severity.upper()}] {f.category} - {f.file}:{f.line}  ({fmt_conf(c.confidence)})")
        print(f"  {f.title}")
        print(f"  why: {f.explanation}")
        print(f"  fix: {f.suggestion}")
        print(f"  verifier: {c.reason}")

    if dropped:
        print(f"\n=== Dropped by verifier ({len(dropped)}) ===")
        for c in dropped:
            f = c.finding
            print(f"\n- {f.file}:{f.line} {f.title}")
            print(f"  [{c.stage}, {fmt_conf(c.confidence)}] {c.reason}")



if __name__ == "__main__":
    main()
