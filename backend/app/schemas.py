"""Typed data shared across the app.

For models the LLM fills in (Finding, Review), the Field descriptions are
not just comments: LangChain copies them into the JSON schema sent to the
model, so they act as instructions per field.
"""

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low"]
Category = Literal["security", "correctness", "performance", "maintainability"]
Specialist = Literal["security", "correctness", "maintainability"]

SEVERITY_RANK: dict[Severity, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class Finding(BaseModel):
    """One problem found in the diff."""

    file: str = Field(description="Path of the file, exactly as it appears in the diff header.")
    line: int = Field(description="Line number in the NEW version of the file (the + side).")
    severity: Severity
    category: Category
    title: str = Field(description="Short summary, under 80 characters.")
    explanation: str = Field(description="Why this is a problem and what can go wrong.")
    suggestion: str = Field(description="Concrete fix, ideally a corrected code snippet.")


class Review(BaseModel):
    """All findings for one diff. An empty list means the diff looks fine."""

    findings: list[Finding]


class TriagePlan(BaseModel):
    """Which specialist reviewers to run for this PR."""

    # `reason` comes first on purpose: the model writes fields in order, so it
    # explains its thinking before committing to a list.
    reason: str = Field(description="One sentence explaining the choice.")
    specialists: list[Specialist] = Field(
        description="Specialists to run. Empty only if there is nothing to review."
    )


class Verdict(BaseModel):
    """The verifier's judgement on one finding."""

    id: int = Field(description="The finding's [id] from the list.")
    # reason before confidence: evidence first, then the score it supports.
    reason: str = Field(
        description="What you checked in the code and what you concluded, in one or two sentences."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "How likely this is a real problem. 0.9+: the code shown clearly has it. "
            "~0.5: plausible but depends on code you cannot see. "
            "Below 0.3: the code contradicts it, or it is speculative."
        ),
    )


class VerifierReport(BaseModel):
    """One verdict per finding."""

    verdicts: list[Verdict]


class FindingCheck(BaseModel):
    """What the verify step decided about one finding, and why (filled by our code)."""

    finding: Finding
    kept: bool
    stage: Literal["grounding", "verifier"]  # which check made the decision
    confidence: float | None  # verifier's score; None if it never got a score
    reason: str


# --- Pull request input (filled by our code, not by the LLM) ---


class ChangedFile(BaseModel):
    """One file the reviewer will look at."""

    path: str
    status: Literal["added", "modified", "renamed"]
    added: int  # number of + lines
    removed: int  # number of - lines
    hunks: list[str]  # each hunk's text, starting with its "@@ -a,b +c,d @@" header


class SkippedFile(BaseModel):
    """A file left out of the review, and why."""

    path: str
    reason: str


class PullRequest(BaseModel):
    url: str
    owner: str
    repo: str
    number: int
    title: str
    description: str
    author: str
    base_ref: str  # branch the PR merges into, e.g. "main"
    head_ref: str  # branch with the changes
    head_sha: str  # exact commit reviewed; Step 14 needs it to post comments
    files: list[ChangedFile]
    skipped: list[SkippedFile]
