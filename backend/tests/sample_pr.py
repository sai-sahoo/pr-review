"""A small, hand-written PR used across the tests: a refund function whose
safety check was deleted. Real diff text goes through the real parser, so the
tests exercise the same code path as a live PR."""

from app.diff import parse_diff
from app.schemas import Finding, PullRequest

PATH = "app/payments.py"

# Lines 11-12 (old file) are removed; nothing replaces the check.
REMOVED_CHECK_DIFF = """\
diff --git a/app/payments.py b/app/payments.py
--- a/app/payments.py
+++ b/app/payments.py
@@ -10,6 +10,5 @@ MAX_REFUND_DAYS = 30
 def refund(order: Order, amount: int) -> int:
-    if amount > order.total:
-        raise ValueError("refund exceeds order total")
+    log.info("refund %s for order %s", amount, order.id)
     order.balance -= amount
     order.save()
     return order.balance
"""

# The whole file after the PR, for read_file to return.
PAYMENTS_PY = """\
import logging

from app.models import Order

log = logging.getLogger(__name__)

MAX_REFUND_DAYS = 30


def refund(order: Order, amount: int) -> int:
    log.info("refund %s for order %s", amount, order.id)
    order.balance -= amount
    order.save()
    return order.balance
"""


def make_pr(diff: str = REMOVED_CHECK_DIFF) -> PullRequest:
    files, skipped = parse_diff(diff)
    return PullRequest(
        url="https://github.com/acme/shop/pull/7", owner="acme", repo="shop", number=7,
        title="Log refunds", description="", author="dev", base_ref="main",
        head_ref="log-refunds", head_sha="abc123", files=files, skipped=skipped,
    )


def finding(**overrides) -> Finding:
    """A valid Finding; tests override only the fields they care about."""
    data = dict(
        file=PATH, line=11, severity="high", category="correctness",
        title="Refund amount is no longer validated",
        explanation="The check against order.total was removed, so refunds can exceed the order.",
        suggestion="Restore: if amount > order.total: raise ValueError(...)",
    )
    return Finding(**(data | overrides))
