"""Minimal GitHub REST client: only what the reviewer needs.

Same PR endpoint, two representations, chosen by the Accept header:
  application/vnd.github+json  -> PR metadata (title, branches, head sha)
  application/vnd.github.diff  -> the raw unified diff as plain text

Reads go out as the GitHub App when it's installed on the repo, and with
GITHUB_TOKEN (or anonymously) otherwise. The one write, post_review, always
needs the App's installation token (app/github_app.py).
"""

import os
import re
from functools import lru_cache

import httpx

from app.diff import DEFAULT_TOKEN_BUDGET, apply_budget, diff_line_numbers, parse_diff
from app.schemas import Finding, PullRequest

API = "https://api.github.com"
PR_URL_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)")


class GitHubError(Exception):
    """A GitHub problem explained in words a user can act on."""


def parse_pr_url(url: str) -> tuple[str, str, int]:
    """'https://github.com/psf/requests/pull/6710' -> ('psf', 'requests', 6710)"""
    match = PR_URL_RE.match(url.strip())
    if not match:
        raise GitHubError(f"Not a GitHub PR URL: {url!r}")
    owner, repo, number = match.groups()
    return owner, repo, int(number)


def read_token(owner: str, repo: str) -> str | None:
    """Who we read owner/repo as.

    The App's installation token when the App is installed there: it can
    read the private repos it's installed on without anyone's personal
    token, and its rate limit is the App's own, not your account's. Else
    GITHUB_TOKEN, for any public PR pasted into the UI. Else None:
    anonymous, 60 requests an hour.
    """
    # Imported here, not at the top: github_app imports this module (for API
    # and GitHubError), and two modules importing each other at the top can
    # find each other half-loaded.
    from app import github_app

    if github_app.configured() and (token := github_app.installation_token(owner, repo)):
        return token
    return os.getenv("GITHUB_TOKEN")


def _client(owner: str, repo: str) -> httpx.Client:
    headers = {"X-GitHub-Api-Version": "2022-11-28"}
    if token := read_token(owner, repo):
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=API, headers=headers, timeout=30)


def _get(
    client: httpx.Client,
    path: str,
    accept: str,
    not_found: str = "PR not found. Check the URL; private repos need the App installed or GITHUB_TOKEN.",
) -> httpx.Response:
    resp = client.get(path, headers={"Accept": accept})
    if resp.status_code == 404:
        raise GitHubError(not_found)
    if resp.status_code in (403, 429):
        raise GitHubError("GitHub refused the request (likely rate limit). Set GITHUB_TOKEN in .env.")
    if resp.status_code == 406:
        raise GitHubError("Diff too large for GitHub's diff endpoint.")
    resp.raise_for_status()  # any other 4xx/5xx becomes an httpx.HTTPStatusError
    return resp


def get_pr_head(url: str) -> tuple[str, str]:
    """(canonical PR URL, head commit sha): one small request, no diff.

    The canonical URL is GitHub's own html_url, the same string a webhook
    sends. So '.../pull/7/files' typed into the UI and the webhook's URL
    become one key, and the dedupe check sees them as the same PR.
    """
    owner, repo, number = parse_pr_url(url)
    with _client(owner, repo) as client:
        meta = _get(client, f"/repos/{owner}/{repo}/pulls/{number}", "application/vnd.github+json").json()
    return meta["html_url"], meta["head"]["sha"]


def get_pull_request(url: str, token_budget: int = DEFAULT_TOKEN_BUDGET) -> PullRequest:
    """Fetch a PR, drop noise, fit it into the budget, return one typed object."""
    owner, repo, number = parse_pr_url(url)
    path = f"/repos/{owner}/{repo}/pulls/{number}"

    with _client(owner, repo) as client:
        meta = _get(client, path, "application/vnd.github+json").json()
        diff_text = _get(client, path, "application/vnd.github.diff").text

    files, skipped = parse_diff(diff_text)
    files, over_budget = apply_budget(files, token_budget)

    return PullRequest(
        url=url,
        owner=owner,
        repo=repo,
        number=number,
        title=meta["title"],
        description=meta.get("body") or "",  # body is null when the PR has no description
        author=meta["user"]["login"],
        base_ref=meta["base"]["ref"],
        head_ref=meta["head"]["ref"],
        head_sha=meta["head"]["sha"],
        files=files,
        skipped=skipped + over_budget,
    )


# Several specialists often read the same file. A commit sha never changes,
# so caching by (owner, repo, sha, path) is always safe.
@lru_cache(maxsize=256)
def get_file_text(owner: str, repo: str, sha: str, path: str) -> str:
    """Full text of one file as of a specific commit."""
    with _client(owner, repo) as client:
        resp = _get(
            client,
            f"/repos/{owner}/{repo}/contents/{path}?ref={sha}",
            "application/vnd.github.raw+json",  # raw file bytes instead of base64 JSON
            not_found=f"{path} does not exist at commit {sha[:7]}",
        )
    return resp.text


# --- posting a review -------------------------------------------------------

SEVERITY_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}


def format_finding(f: Finding) -> str:
    """One finding as Markdown, the way it reads on the PR."""
    return (
        f"{SEVERITY_ICON[f.severity]} **{f.severity} · {f.category}: {f.title}**\n\n"
        f"{f.explanation}\n\n**Suggestion:** {f.suggestion}"
    )


def review_payload(pr: PullRequest, findings: list[Finding]) -> dict:
    """The body for POST /repos/{o}/{r}/pulls/{n}/reviews.

    An inline comment is placed with (path, line, side). "RIGHT" is the new
    version of the file, and `line` is a line number in it: exactly what
    Finding.line holds. GitHub only accepts lines that appear in the diff
    (+ or context lines), and a single bad comment fails the whole request
    with a 422. The verifier already drops findings outside the diff; this
    check makes sure, and moves any such finding into the summary instead.
    """
    visible = {f.path: diff_line_numbers(f.hunks) for f in pr.files}
    comments, general = [], []
    for f in findings:
        if f.line in visible.get(f.file, ()):
            comments.append({"path": f.file, "line": f.line, "side": "RIGHT", "body": format_finding(f)})
        else:
            general.append(f"`{f.file}:{f.line}`: {format_finding(f)}")

    if findings:
        summary = f"**pr-review** found {len(findings)} issue{'s' * (len(findings) != 1)}."
    else:
        summary = "**pr-review** found no issues. ✅"
    return {
        # The commit the worker actually reviewed. If someone pushed since,
        # GitHub still pins the comments to this commit and marks them
        # "outdated" where the lines changed, instead of putting them on the
        # wrong lines.
        "commit_id": pr.head_sha,
        # COMMENT: give feedback without approving or blocking the merge.
        # A bot shouldn't use REQUEST_CHANGES; that's for a human to decide.
        "event": "COMMENT",
        "body": "\n\n---\n\n".join([summary, *general]),
        "comments": comments,
    }


def post_review(pr: PullRequest, payload: dict, token: str) -> str:
    """Post one review with all its inline comments. Returns its URL on GitHub."""
    with httpx.Client(base_url=API, timeout=30, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }) as client:
        resp = client.post(f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/reviews", json=payload)
    if resp.status_code == 403:
        raise GitHubError("GitHub refused to post the review: the App needs Pull requests: Read and write.")
    if resp.status_code == 422:  # e.g. a comment on a line outside the diff
        raise GitHubError(f"GitHub rejected the review: {resp.json().get('errors') or resp.text}")
    resp.raise_for_status()
    return resp.json()["html_url"]
