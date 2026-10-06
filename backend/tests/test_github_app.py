"""Step 14c: authenticating as the GitHub App, building the review, posting it.

No network: httpx.MockTransport answers requests with a Python function, so
we can see exactly what we sent and decide what GitHub "replies".
"""

import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app import github_app
from app.github_app import app_jwt, installation_token
from app.github_client import GitHubError, review_payload
from app.graph.nodes import publish
from sample_pr import PATH, finding, make_pr

APP_ID = "123456"


@pytest.fixture(scope="module")
def rsa_key():
    """A throwaway key pair, like the .pem GitHub generates (and the public half it keeps)."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def app_env(monkeypatch, tmp_path, rsa_key):
    """Configure the App the way .env does, with the private key in a file."""
    pem = rsa_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    path = tmp_path / "app.pem"
    path.write_bytes(pem)
    monkeypatch.setenv("GITHUB_APP_ID", APP_ID)
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", str(path))
    monkeypatch.setattr(github_app, "_tokens", {})  # no tokens cached by another test


@pytest.fixture
def fake_github_api(monkeypatch):
    """Route the App's HTTP calls to `handler`. Returns the list of requests seen."""
    requests: list[httpx.Request] = []
    replies: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return replies[f"{request.method} {request.url.path}"]

    def client(auth: str) -> httpx.Client:
        return httpx.Client(base_url="https://api.github.com", headers={"Authorization": auth},
                            transport=httpx.MockTransport(handler))

    monkeypatch.setattr(github_app, "_client", client)
    return requests, replies


def test_app_jwt_is_signed_by_the_private_key(app_env, rsa_key):
    token = app_jwt()

    # GitHub's side: verify with the public key. A wrong key or a changed
    # claim would raise here.
    claims = jwt.decode(token, rsa_key.public_key(), algorithms=["RS256"])
    assert claims["iss"] == APP_ID
    assert claims["iat"] < time.time() < claims["exp"]
    assert claims["exp"] - claims["iat"] <= 10 * 60  # GitHub's limit


def test_installation_token_exchange_and_cache(app_env, rsa_key, fake_github_api):
    requests, replies = fake_github_api
    replies["GET /repos/acme/shop/installation"] = httpx.Response(200, json={"id": 42})
    replies["POST /app/installations/42/access_tokens"] = httpx.Response(
        201, json={"token": "ghs_abc", "expires_at": "2099-01-01T00:00:00Z"})

    assert installation_token("acme", "shop") == "ghs_abc"
    assert installation_token("acme", "shop") == "ghs_abc"  # from the cache

    assert len(requests) == 2  # one exchange for both calls
    # Both App endpoints were called with the JWT, not with a token.
    auth = requests[0].headers["Authorization"].removeprefix("Bearer ")
    assert jwt.decode(auth, rsa_key.public_key(), algorithms=["RS256"])["iss"] == APP_ID
    assert requests[1].read() == b'{"repositories":["shop"]}'  # narrowed to one repo


def test_expiring_token_is_replaced(app_env, fake_github_api):
    requests, replies = fake_github_api
    soon = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 60))  # inside the margin
    replies["GET /repos/acme/shop/installation"] = httpx.Response(200, json={"id": 42})
    replies["POST /app/installations/42/access_tokens"] = httpx.Response(
        201, json={"token": "ghs_old", "expires_at": soon})

    installation_token("acme", "shop")
    installation_token("acme", "shop")

    assert len(requests) == 4  # the first token was too close to expiry to reuse


def test_app_not_installed_on_the_repo_is_none(app_env, fake_github_api):
    _, replies = fake_github_api
    replies["GET /repos/alexreardon/tiny-invariant/installation"] = httpx.Response(404)

    assert installation_token("alexreardon", "tiny-invariant") is None


def test_wrong_app_id_or_key_is_explained(app_env, fake_github_api):
    _, replies = fake_github_api
    replies["GET /repos/acme/shop/installation"] = httpx.Response(401)

    with pytest.raises(GitHubError, match="GITHUB_APP_ID"):
        installation_token("acme", "shop")


# --- the review payload ----------------------------------------------------


def test_findings_on_diff_lines_become_inline_comments():
    pr = make_pr()
    on_diff = finding(line=11, severity="critical")
    off_diff = finding(line=2, severity="low", title="Unused import")  # line 2 isn't in any hunk

    payload = review_payload(pr, [on_diff, off_diff])

    assert payload["commit_id"] == pr.head_sha
    assert payload["event"] == "COMMENT"
    [comment] = payload["comments"]
    assert (comment["path"], comment["line"], comment["side"]) == (PATH, 11, "RIGHT")
    assert "critical · correctness: Refund amount is no longer validated" in comment["body"]
    # The one GitHub would reject goes into the summary, so it isn't lost.
    assert payload["body"].startswith("**pr-review** found 2 issues.")
    assert f"`{PATH}:2`" in payload["body"] and "Unused import" in payload["body"]


def test_no_findings_still_reports_back():
    payload = review_payload(make_pr(), [])

    assert payload["comments"] == []
    assert "no issues" in payload["body"]


# --- the publish node ------------------------------------------------------


def state(*findings) -> dict:
    return {"pr": make_pr(), "verified": list(findings)}


def test_publish_without_an_app_posts_nothing():
    # conftest leaves the App unconfigured; the seams would fail the test if called.
    assert publish(state(finding())) == {"review_url": None, "publish_note": "not posted: no GitHub App configured"}


def test_publish_posts_as_the_installation(app_env, monkeypatch):
    posted = []
    monkeypatch.setattr("app.graph.nodes.installation_token", lambda owner, repo: "ghs_abc")
    monkeypatch.setattr("app.graph.nodes.post_review",
                        lambda pr, payload, token: posted.append((payload, token)) or "https://github.com/r/1")

    update = publish(state(finding()))

    assert update == {"review_url": "https://github.com/r/1", "publish_note": "posted"}
    [(payload, token)] = posted
    assert token == "ghs_abc" and len(payload["comments"]) == 1


def test_publish_skips_repos_without_the_app(app_env, monkeypatch):
    monkeypatch.setattr("app.graph.nodes.installation_token", lambda owner, repo: None)

    assert publish(state())["publish_note"] == "not posted: the App isn't installed on acme/shop"


def test_a_failed_post_does_not_fail_the_review(app_env, monkeypatch):
    def refused(pr, payload, token):
        raise GitHubError("GitHub refused to post the review: the App needs Pull requests: Read and write.")

    monkeypatch.setattr("app.graph.nodes.installation_token", lambda owner, repo: "ghs_abc")
    monkeypatch.setattr("app.graph.nodes.post_review", refused)

    update = publish(state(finding()))  # returns instead of raising

    assert update["review_url"] is None
    assert update["publish_note"].startswith("could not post: GitHub refused")
