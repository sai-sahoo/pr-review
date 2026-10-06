"""Authenticate as our GitHub App, so reviews are posted under the App's name.

A GitHub App has two identities, and you need the first to get the second:

  1. The App itself. We prove it with a JWT signed by the App's private key
     (the .pem). GitHub checks it with the public half, which it kept when it
     generated the key. Lives 10 minutes at most. It can only do App-level
     things, like "which installation covers this repo?" and "give me a token".
  2. An installation: the App installed on one account or set of repos. Its
     token (ghs_...) works like a personal token, limited to the installed
     repos and the permissions you ticked. Lives 1 hour.

    private key --sign--> JWT --GET /repos/{o}/{r}/installation--> installation id
                              --POST /app/installations/{id}/access_tokens--> token
"""

import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

import httpx
import jwt

from app.github_client import API, GitHubError

TOKEN_REFRESH_MARGIN = 5 * 60  # seconds; get a new token this long before the old one expires

# (owner, repo) -> (token, expires at as a Unix timestamp). Specialists run in
# threads, hence the lock. The worker is one long-lived process, so one token
# serves every review of that repo for most of an hour.
_tokens: dict[tuple[str, str], tuple[str, float]] = {}
_tokens_lock = Lock()


def configured() -> bool:
    """True when .env names an App. Without one, reviews just aren't posted."""
    return bool(os.getenv("GITHUB_APP_ID") and os.getenv("GITHUB_APP_PRIVATE_KEY_PATH"))


def app_jwt() -> str:
    """A JWT that says "I am App <id>", signed with the App's private key."""
    now = int(time.time())
    claims = {
        "iss": os.environ["GITHUB_APP_ID"],  # issuer: which App this is
        # Issued 60 s in the past, in case this machine's clock runs a little
        # ahead of GitHub's: a token from "the future" is rejected.
        "iat": now - 60,
        "exp": now + 9 * 60,  # GitHub's maximum is 10 minutes; leave a margin
    }
    key = Path(os.environ["GITHUB_APP_PRIVATE_KEY_PATH"]).expanduser().read_text()
    # RS256: an RSA signature. Only the private key can make it; GitHub
    # verifies it with the public key. Unlike the webhook's HMAC, the two
    # sides don't share a secret.
    return jwt.encode(claims, key, algorithm="RS256")


def _client(auth: str) -> httpx.Client:
    """The App endpoints' client. Tests replace it with one on a fake transport."""
    return httpx.Client(
        base_url=API,
        headers={
            "Authorization": auth,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30,
    )


def installation_token(owner: str, repo: str) -> str | None:
    """A token for acting on owner/repo as the App, or None if the App isn't
    installed there (e.g. a public repo you only review from the UI).
    """
    with _tokens_lock:
        cached = _tokens.get((owner, repo))
        if cached and cached[1] - time.time() > TOKEN_REFRESH_MARGIN:
            return cached[0]

        with _client(f"Bearer {app_jwt()}") as client:
            resp = client.get(f"/repos/{owner}/{repo}/installation")
            if resp.status_code == 404:
                return None
            if resp.status_code == 401:
                raise GitHubError("GitHub rejected the App's JWT. Check GITHUB_APP_ID and the private key.")
            resp.raise_for_status()
            installation_id = resp.json()["id"]

            # "repositories" narrows the token to this one repo, even if the
            # App is installed on more: a leaked token can do less damage.
            resp = client.post(f"/app/installations/{installation_id}/access_tokens",
                               json={"repositories": [repo]})
            resp.raise_for_status()
            data = resp.json()

        expires_at = datetime.fromisoformat(data["expires_at"]).timestamp()  # "2026-10-06T12:00:00Z"
        _tokens[(owner, repo)] = (data["token"], expires_at)
        return data["token"]
