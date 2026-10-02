"""Turn a raw unified diff into ChangedFile objects the reviewer can use.

Two jobs:
1. Filter out noise (lockfiles, binaries, deletions) that costs tokens
   but has nothing worth reviewing.
2. Keep the rest under a token budget so the prompt fits the model.
"""

import re

from unidiff import PatchedFile, PatchSet

from app.schemas import ChangedFile, SkippedFile

DEFAULT_TOKEN_BUDGET = 30_000

# "@@ -10,7 +12,9 @@ def foo():" -> captures 12, where the hunk starts in the NEW file
HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)")

# Generated files: huge diffs, never hand-written, nothing to review.
IGNORED_FILENAMES = {"package-lock.json", "pnpm-lock.yaml", "go.sum"}
IGNORED_SUFFIXES = (".lock", ".min.js", ".min.css", ".map", ".svg")


def estimate_tokens(text: str) -> int:
    """Rough count: about 4 characters per token. Good enough for budgeting."""
    return len(text) // 4


def file_tokens(f: ChangedFile) -> int:
    return estimate_tokens("".join(f.hunks))


def _skip_reason(pf: PatchedFile) -> str | None:
    name = pf.path.rsplit("/", 1)[-1]
    if pf.is_removed_file:
        return "deleted"
    if pf.is_binary_file:
        return "binary"
    if name in IGNORED_FILENAMES or name.endswith(IGNORED_SUFFIXES):
        return "lockfile/generated"
    if len(pf) == 0:  # a PatchedFile is a list of hunks; none = pure rename or mode change
        return "no content changes"
    return None


def _status(pf: PatchedFile) -> str:
    if pf.is_added_file:
        return "added"
    if pf.is_rename:
        return "renamed"
    return "modified"


def parse_diff(diff_text: str) -> tuple[list[ChangedFile], list[SkippedFile]]:
    """Split a unified diff into reviewable files and skipped ones."""
    files: list[ChangedFile] = []
    skipped: list[SkippedFile] = []
    for pf in PatchSet(diff_text):
        reason = _skip_reason(pf)
        if reason:
            skipped.append(SkippedFile(path=pf.path, reason=reason))
            continue
        files.append(
            ChangedFile(
                path=pf.path,
                status=_status(pf),
                added=pf.added,
                removed=pf.removed,
                hunks=[str(hunk) for hunk in pf],
            )
        )
    return files, skipped


def _walk_hunk(hunk: str):
    """Yield (new_line_number, line) for each line after the header.

    '+' and ' ' lines exist in the new file and get a number; '-' lines
    (and "\\ No newline at end of file") don't, so they get None.
    """
    header, *lines = hunk.splitlines()
    new_line = int(HUNK_HEADER_RE.match(header).group(1))
    for line in lines:
        if line.startswith(("-", "\\")):
            yield None, line
        else:
            yield new_line, line
            new_line += 1


def number_hunk(hunk: str) -> str:
    """Prefix each line with its line number in the NEW file.

    LLMs are bad at counting lines, and Finding.line must be a new-file
    line number, so we compute it in code and show it in the prompt.
    """
    out = [hunk.splitlines()[0]]  # the @@ header, unchanged
    for n, line in _walk_hunk(hunk):
        out.append(f"     {line}" if n is None else f"{n:>4} {line}")
    return "\n".join(out)


def diff_line_numbers(hunks: list[str]) -> set[int]:
    """New-file line numbers visible in these hunks ('+' and context lines).

    A finding must point at one of these: it's what the reviewer was shown,
    and the only lines GitHub accepts inline comments on.
    """
    return {n for h in hunks for n, _ in _walk_hunk(h) if n is not None}


def apply_budget(
    files: list[ChangedFile], max_tokens: int
) -> tuple[list[ChangedFile], list[SkippedFile]]:
    """Keep files in diff order while they fit; skip any file that doesn't.

    A skipped big file doesn't stop the loop, so smaller files after it
    can still fit in the remaining budget.
    """
    kept: list[ChangedFile] = []
    skipped: list[SkippedFile] = []
    used = 0
    for f in files:
        cost = file_tokens(f)
        if used + cost > max_tokens:
            skipped.append(SkippedFile(path=f.path, reason=f"over token budget (~{cost} tokens)"))
            continue
        kept.append(f)
        used += cost
    return kept, skipped
