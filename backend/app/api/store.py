"""An in-memory home for reviews: a dict behind a lock.

Good enough to learn the API shape. Everything is lost on restart, and two
server processes would each have their own dict. Step 12 swaps this for
Postgres behind the same create/get/update methods.
"""

import threading
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas import Finding, FindingCheck

Status = Literal["queued", "running", "done", "failed"]


def _now() -> datetime:
    return datetime.now(UTC)


class ReviewRecord(BaseModel):
    """One review job, as the API returns it."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    pr_url: str
    status: Status = "queued"
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    title: str | None = None  # the PR title, once fetched
    findings: list[Finding] = []  # the verified ones: the final answer
    checks: list[FindingCheck] = []  # every finding with the verifier's decision
    error: str | None = None  # set when status == "failed"


class ReviewStore:
    def __init__(self) -> None:
        self._records: dict[str, ReviewRecord] = {}
        # The review runs in a worker thread while the event loop serves GETs.
        # The lock makes each read or write happen as a whole, never half-done.
        self._lock = threading.Lock()

    def create(self, pr_url: str) -> ReviewRecord:
        record = ReviewRecord(pr_url=pr_url)
        with self._lock:
            self._records[record.id] = record
        return record

    def get(self, review_id: str) -> ReviewRecord | None:
        with self._lock:
            return self._records.get(review_id)

    def list(self) -> list[ReviewRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def update(self, review_id: str, **changes) -> ReviewRecord:
        # Replace the record with a changed copy instead of mutating it, so a
        # reader holding the old object never sees it change under its feet.
        with self._lock:
            record = self._records[review_id].model_copy(update=changes)
            self._records[review_id] = record
        return record
