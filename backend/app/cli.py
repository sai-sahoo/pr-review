"""Step 3: fetch a real PR and show exactly what the reviewer will see.

Run:  cd backend && uv run python -m app.cli https://github.com/OWNER/REPO/pull/123
      add --hunks to print the full diff text, --budget N to change the token limit
"""

import argparse

from app.diff import DEFAULT_TOKEN_BUDGET, file_tokens
from app.github_client import GitHubError, get_pull_request


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a GitHub PR and print its reviewable diff.")
    parser.add_argument("pr_url")
    parser.add_argument("--budget", type=int, default=DEFAULT_TOKEN_BUDGET, help="max diff tokens to keep")
    parser.add_argument("--hunks", action="store_true", help="print full hunk text, not just headers")
    args = parser.parse_args()

    try:
        pr = get_pull_request(args.pr_url, token_budget=args.budget)
    except GitHubError as e:
        raise SystemExit(f"error: {e}")

    print(f"#{pr.number} {pr.title}")
    print(f"by {pr.author}: {pr.head_ref} -> {pr.base_ref} @ {pr.head_sha[:7]}")

    print(f"\n=== Reviewable files ({len(pr.files)}) ===")
    for f in pr.files:
        print(f"{f.status:<9} {f.path}  +{f.added} -{f.removed}  ~{file_tokens(f)} tokens")
        for hunk in f.hunks:
            # The first line of a hunk is its header: "@@ -old_start,len +new_start,len @@"
            print(hunk if args.hunks else f"    {hunk.splitlines()[0]}")

    print(f"\n=== Skipped ({len(pr.skipped)}) ===")
    for s in pr.skipped:
        print(f"{s.path}  ({s.reason})")

    total = sum(file_tokens(f) for f in pr.files)
    print(f"\nTotal: ~{total} / {args.budget} tokens")


if __name__ == "__main__":
    main()
