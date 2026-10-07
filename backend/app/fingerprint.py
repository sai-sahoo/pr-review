"""Recognise "the same finding" across reviews of one PR.

Each push gets a fresh review, and most of what it finds was already found
last time. To skip those we need an identity for a finding that stays the
same from one review to the next:

  - not the line number: add 3 lines above the bug and it moves from 42 to 45
  - not the title or explanation: the LLM words the same problem differently every run
  - not the category: one run calls it "security", the next "correctness"

What does stay the same is the code the finding points at. So:

    fingerprint = hash(file path + the code on that line, whitespace squeezed)

Aggregate already keeps one finding per line, so one fingerprint per line
matches that rule. If the line itself is edited, the fingerprint changes and
the finding counts as new: the code is different now, so asking again is fair.

Known limit: the same code twice in one file (two identical `return None`
lines) gives one fingerprint, so a finding on the second could be skipped.

The fingerprint travels inside each posted comment as a hidden marker,
an HTML comment GitHub doesn't render:

    <!-- pr-review:3f2a9c01b7de -->

so the next review can read back what's on the PR without our database.
"""

import hashlib
import re

from app.diff import line_text
from app.schemas import Finding, PullRequest

MARKER_RE = re.compile(r"<!-- pr-review:([0-9a-f]{12}) -->")


def code_fingerprint(path: str, code: str) -> str:
    """The fingerprint of one line of code in one file."""
    squeezed = " ".join(code.split())
    return hashlib.sha1(f"{path}\n{squeezed}".encode()).hexdigest()[:12]


def fingerprint(pr: PullRequest, finding: Finding) -> str:
    hunks = next((f.hunks for f in pr.files if f.path == finding.file), [])
    code = line_text(hunks, finding.line)
    # Verified findings are always on a diff line (the grounding check). The
    # line number is only a fallback for one that somehow isn't.
    return code_fingerprint(finding.file, code if code is not None else f"line {finding.line}")


def marker(fp: str) -> str:
    return f"<!-- pr-review:{fp} -->"
