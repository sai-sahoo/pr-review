"""GitHub webhooks: prove a request really came from GitHub, then read the
few fields of a pull_request event that we need.

How GitHub signs a delivery: it takes the raw request body and the secret
you typed into the App settings, computes HMAC-SHA256(secret, body), and
sends it as the header  X-Hub-Signature-256: sha256=<hex>.  We do the same
computation on our side. Only someone who knows the secret can produce a
matching value, and changing a single byte of the body changes it completely.
"""

import hashlib
import hmac

from pydantic import BaseModel

# The actions that mean "there is new code to review". Everything else
# (closed, edited, labeled, assigned, review_requested, ...) is ignored.
#   opened            a new PR
#   synchronize       new commits were pushed to it
#   reopened          a closed PR is open again
#   ready_for_review  a draft became a real PR
REVIEW_ACTIONS = {"opened", "synchronize", "reopened", "ready_for_review"}


def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    """True if `header` is GitHub's signature of exactly these body bytes."""
    if not header:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    # compare_digest takes the same time however many leading characters
    # match. A plain == stops at the first difference, and by timing many
    # guesses an attacker could find the right value one character at a time.
    return hmac.compare_digest(expected.encode(), header.encode())


# A pull_request payload is a few hundred fields. These models name only the
# ones we read; Pydantic ignores the rest.
class Head(BaseModel):
    sha: str


class PullRequestInfo(BaseModel):
    html_url: str  # the same canonical URL get_pr_head returns, so dedupe matches
    draft: bool = False
    head: Head


class PullRequestEvent(BaseModel):
    action: str
    pull_request: PullRequestInfo

    def skip_reason(self) -> str | None:
        """Why this event shouldn't start a review, or None if it should."""
        if self.action not in REVIEW_ACTIONS:
            return f"action {self.action!r}"
        if self.pull_request.draft:
            # Drafts change a lot and nobody asked for a review yet. When it's
            # marked ready, the ready_for_review event starts one.
            return "draft PR"
        return None
