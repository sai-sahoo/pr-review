"""Plain functions, no LLM: the cheapest and most reliable tests come first."""

from app.diff import diff_line_numbers, number_hunk, parse_diff
from sample_pr import PATH, REMOVED_CHECK_DIFF

EXTRA_FILES = """\
diff --git a/uv.lock b/uv.lock
--- a/uv.lock
+++ b/uv.lock
@@ -1 +1 @@
-version = 1
+version = 2
diff --git a/app/legacy.py b/app/legacy.py
deleted file mode 100644
--- a/app/legacy.py
+++ /dev/null
@@ -1,2 +0,0 @@
-x = 1
-y = 2
"""


def test_removed_lines_get_no_line_number():
    hunk = parse_diff(REMOVED_CHECK_DIFF)[0][0].hunks[0]
    # Exactly what the specialists see in their prompt.
    assert number_hunk(hunk).splitlines()[1:] == [
        "  10  def refund(order: Order, amount: int) -> int:",
        "     -    if amount > order.total:",
        '     -        raise ValueError("refund exceeds order total")',
        '  11 +    log.info("refund %s for order %s", amount, order.id)',
        "  12      order.balance -= amount",
        "  13      order.save()",
        "  14      return order.balance",
    ]


def test_diff_line_numbers_are_new_file_lines_only():
    hunks = parse_diff(REMOVED_CHECK_DIFF)[0][0].hunks
    assert diff_line_numbers(hunks) == {10, 11, 12, 13, 14}


def test_parse_diff_skips_lockfiles_and_deleted_files():
    files, skipped = parse_diff(REMOVED_CHECK_DIFF + EXTRA_FILES)
    assert [f.path for f in files] == [PATH]
    assert [(s.path, s.reason) for s in skipped] == [
        ("uv.lock", "lockfile/generated"),
        ("app/legacy.py", "deleted"),
    ]
