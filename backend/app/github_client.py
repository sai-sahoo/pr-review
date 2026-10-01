"""Minimal GitHub REST client: only what the reviewer needs.

Same PR endpoint, two representations, chosen by the Accept header:
  application/vnd.github+json  -> PR metadata (title, branches, head sha)
  application/vnd.github.diff  -> the raw unified diff as plain text
"""

import os
import re

import httpx

from app.diff import DEFAULT_TOKEN_BUDGET, apply_budget, parse_diff
from app.schemas import PullRequest

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


def _client() -> httpx.Client:
    headers = {"X-GitHub-Api-Version": "2022-11-28"}
    token = os.getenv("GITHUB_TOKEN")
    if token:  # optional for public repos, but lifts the limit from 60 to 5000 requests/hour
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=API, headers=headers, timeout=30)


def _get(client: httpx.Client, path: str, accept: str) -> httpx.Response:
    resp = client.get(path, headers={"Accept": accept})
    if resp.status_code == 404:
        raise GitHubError("PR not found. Check the URL; private repos need GITHUB_TOKEN.")
    if resp.status_code in (403, 429):
        raise GitHubError("GitHub refused the request (likely rate limit). Set GITHUB_TOKEN in .env.")
    if resp.status_code == 406:
        raise GitHubError("Diff too large for GitHub's diff endpoint.")
    resp.raise_for_status()  # any other 4xx/5xx becomes an httpx.HTTPStatusError
    return resp


def get_pull_request(url: str, token_budget: int = DEFAULT_TOKEN_BUDGET) -> PullRequest:
    """Fetch a PR, drop noise, fit it into the budget, return one typed object."""
    owner, repo, number = parse_pr_url(url)
    path = f"/repos/{owner}/{repo}/pulls/{number}"

    with _client() as client:
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
