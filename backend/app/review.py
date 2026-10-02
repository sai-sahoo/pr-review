"""Review a real PR with a LangGraph:
    fetch_pr -> triage -> specialists in parallel -> aggregate

Run:  cd backend && uv run python -m app.review https://github.com/OWNER/REPO/pull/123
      add --show-graph to print the graph as a Mermaid diagram (no API calls)
"""

import argparse
import time

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
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a GitHub PR with an LLM.")
    parser.add_argument("pr_url", nargs="?")
    parser.add_argument("--show-graph", action="store_true", help="print the graph and exit")
    args = parser.parse_args()

    graph = build_graph()

    if args.show_graph:
        print(graph.get_graph().draw_mermaid())  # paste into https://mermaid.live
        return
    if not args.pr_url:
        parser.error("pr_url is required")

    start = time.perf_counter()
    findings = []
    try:
        # stream_mode="updates" yields {node_name: update} each time a node finishes,
        # so we can watch the graph run. invoke() would only return the end state.
        for chunk in graph.stream({"pr_url": args.pr_url}, stream_mode="updates"):
            for node, update in chunk.items():
                print(f"[{time.perf_counter() - start:5.1f}s] {node:<10} {describe(node, update)}")
                if node == "aggregate":
                    findings = update["findings"]
    except GitHubError as e:  # node exceptions propagate out of the graph unchanged
        raise SystemExit(f"error: {e}")

    print(f"\n=== Findings ({len(findings)}) ===")
    for f in findings:  # aggregate already sorted them by severity
        print(f"\n[{f.severity.upper()}] {f.category} - {f.file}:{f.line}")
        print(f"  {f.title}")
        print(f"  why: {f.explanation}")
        print(f"  fix: {f.suggestion}")


if __name__ == "__main__":
    main()
