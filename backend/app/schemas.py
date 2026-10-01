"""Typed shapes the agents must return.

The Field descriptions are not just comments: LangChain copies them into
the JSON schema sent to the model, so they act as instructions per field.
"""

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low"]
Category = Literal["security", "correctness", "performance", "maintainability"]


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
