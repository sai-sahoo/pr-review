"""Step 4: review a real PR with a LangGraph:  fetch_pr -> review -> END

Run:  cd backend && uv run python -m app.review https://github.com/OWNER/REPO/pull/123
      add --show-graph to print the graph as a Mermaid diagram (no API calls)
"""

import argparse

from app.github_client import GitHubError
from app.graph import build_graph

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


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

    try:
        # In: the initial state. Out: the final state after the graph reaches END.
        final = graph.invoke({"pr_url": args.pr_url})
    except GitHubError as e:  # node exceptions propagate out of invoke() unchanged
        raise SystemExit(f"error: {e}")

    pr, findings = final["pr"], final["findings"]
    print(f"#{pr.number} {pr.title}")
    print(f"reviewed {len(pr.files)} files, skipped {len(pr.skipped)}")
    print(f"\n=== Findings ({len(findings)}) ===")
    for f in sorted(findings, key=lambda f: SEVERITY_ORDER[f.severity]):
        print(f"\n[{f.severity.upper()}] {f.category} - {f.file}:{f.line}")
        print(f"  {f.title}")
        print(f"  why: {f.explanation}")
        print(f"  fix: {f.suggestion}")


if __name__ == "__main__":
    main()
