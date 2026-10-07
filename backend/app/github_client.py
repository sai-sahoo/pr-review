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
from dataclasses import dataclass
from functools import lru_cache

import httpx

from app.diff import DEFAULT_TOKEN_BUDGET, apply_budget, diff_line_numbers, parse_diff
from app.fingerprint import MARKER_RE, fingerprint, marker
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


def get_posted_fingerprints(pr: PullRequest) -> set[str]:
    """Fingerprints of the findings our bot has already commented on this PR.

    Reads every inline review comment (all reviews, all commits), keeps the
    ones written by a bot, and collects the hidden markers in them. The PR
    itself is the record: a comment someone deleted is gone from here too,
    and one a human marked resolved is still here, so it's not raised again.
    """
    found: set[str] = set()
    with _client(pr.owner, pr.repo) as client:
        url = f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/comments?per_page=100"
        # GitHub returns at most 100 per page. The Link response header points
        # to the next page; httpx parses it into resp.links. No "next" = last page.
        while url:
            resp = _get(client, url, "application/vnd.github+json")
            for comment in resp.json():
                # Only bots: a human quoting our comment in a reply copies the marker too.
                if comment["user"]["type"] == "Bot":
                    found.update(MARKER_RE.findall(comment["body"]))
            url = resp.links.get("next", {}).get("url")
    return found


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
        # The hidden marker lets the next review recognise this finding (app/fingerprint.py).
        body = f"{format_finding(f)}\n\n{marker(fingerprint(pr, f))}"
        if f.line in visible.get(f.file, ()):
            comments.append({"path": f.file, "line": f.line, "side": "RIGHT", "body": body})
        else:
            # get_posted_fingerprints only reads inline comments, so one of
            # these would be posted again next time. Rare: the verifier's
            # grounding check already drops findings outside the diff.
            general.append(f"`{f.file}:{f.line}`: {body}")

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


def _app_client(token: str) -> httpx.Client:
    """A client acting as the App's installation. Tests replace it with a fake transport."""
    return httpx.Client(base_url=API, timeout=30, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })


def post_review(pr: PullRequest, payload: dict, token: str) -> str:
    """Post one review with all its inline comments. Returns its URL on GitHub."""
    with _app_client(token) as client:
        resp = client.post(f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/reviews", json=payload)
    if resp.status_code == 403:
        raise GitHubError("GitHub refused to post the review: the App needs Pull requests: Read and write.")
    if resp.status_code == 422:  # e.g. a comment on a line outside the diff
        raise GitHubError(f"GitHub rejected the review: {resp.json().get('errors') or resp.text}")
    resp.raise_for_status()
    return resp.json()["html_url"]


# --- resolving fixed threads (GraphQL) ---------------------------------------
#
# REST has no notion of a review *thread* (a comment plus its replies) or of
# "resolved". Only the GraphQL API has them. GraphQL is one endpoint,
# POST /graphql, and the request says exactly which fields it wants back.

OPEN_THREADS_QUERY = """
query($owner: String!, $repo: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          path
          comments(first: 1) { nodes { body author { __typename } } }
        }
      }
    }
  }
}
"""

RESOLVE_THREAD_MUTATION = """
mutation($id: ID!) {
  resolveReviewThread(input: {threadId: $id}) { thread { isResolved } }
}
"""


@dataclass
class ReviewThread:
    """One open thread our bot started, and the finding it's about."""

    id: str  # GraphQL node id, what resolveReviewThread takes
    path: str
    fingerprint: str


def graphql(token: str, query: str, variables: dict) -> dict:
    """Run one GraphQL query or mutation and return its "data".

    GraphQL reports most problems with HTTP 200 and an "errors" list in the
    body, so checking the status code alone would miss them.
    """
    with _app_client(token) as client:
        resp = client.post("/graphql", json={"query": query, "variables": variables})
    resp.raise_for_status()
    body = resp.json()
    if body.get("errors"):
        raise GitHubError(f"GitHub GraphQL error: {'; '.join(e['message'] for e in body['errors'])}")
    return body["data"]


def get_open_bot_threads(pr: PullRequest, token: str) -> list[ReviewThread]:
    """The unresolved threads on this PR that start with one of our marked comments."""
    threads: list[ReviewThread] = []
    after = None  # GraphQL pages with a cursor: "the 100 after this one"
    while True:
        data = graphql(token, OPEN_THREADS_QUERY,
                       {"owner": pr.owner, "repo": pr.repo, "number": pr.number, "after": after})
        page = data["repository"]["pullRequest"]["reviewThreads"]
        for t in page["nodes"]:
            first = t["comments"]["nodes"][0] if t["comments"]["nodes"] else None
            if t["isResolved"] or first is None or (first["author"] or {}).get("__typename") != "Bot":
                continue
            if match := MARKER_RE.search(first["body"]):
                threads.append(ReviewThread(id=t["id"], path=t["path"], fingerprint=match.group(1)))
        if not page["pageInfo"]["hasNextPage"]:
            return threads
        after = page["pageInfo"]["endCursor"]


def resolve_thread(token: str, thread_id: str) -> None:
    """Mark a thread "Resolved", as the App. It stays on the PR, collapsed,
    and anyone can unresolve it with one click.
    """
    graphql(token, RESOLVE_THREAD_MUTATION, {"id": thread_id})
