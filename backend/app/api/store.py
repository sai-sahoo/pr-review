"""An in-memory home for reviews: a dict behind a lock, plus each review's events.

Good enough to learn the API shape. Everything is lost on restart, and two
server processes would each have their own dict. Step 12 swaps this for
Postgres behind the same create/get/update methods, and Step 13 moves the
events to Redis pub/sub.
"""

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.api.events import TERMINAL
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
        # GET handlers read records while the review writes them, so the lock
        # makes each read or write happen as a whole, never half-done.
        self._lock = threading.Lock()

        # Each review's events, in order. A list, not a queue: a client that
        # connects late (or reconnects) replays from any position.
        self._events: dict[str, list[dict[str, Any]]] = {}
        # Wakes up the SSE handlers waiting for a review's next event.
        self._new_event: dict[str, asyncio.Condition] = {}

    def create(self, pr_url: str) -> ReviewRecord:
        record = ReviewRecord(pr_url=pr_url)
        with self._lock:
            self._records[record.id] = record
            self._events[record.id] = []
            self._new_event[record.id] = asyncio.Condition()
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

    # --- events ---------------------------------------------------------
    # Only ever called from the event loop: asyncio.Condition is not thread-safe.

    async def add_event(self, review_id: str, event: dict[str, Any]) -> None:
        self._events[review_id].append(event)
        condition = self._new_event[review_id]
        async with condition:  # notify_all() requires holding the condition's lock
            condition.notify_all()

    async def follow(self, review_id: str, start: int = 0) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        """Yield (index, event) from `start` on: first the stored ones, then live
        ones as they arrive. Ends after the done/failed event.
        """
        events = self._events[review_id]
        condition = self._new_event[review_id]

        def finished() -> bool:
            return bool(events) and events[-1]["type"] in TERMINAL

        i = start
        while True:
            async with condition:
                # wait() releases the lock while asleep, so add_event can get in.
                # wait_for re-checks the predicate after every wake-up.
                await condition.wait_for(lambda: len(events) > i or finished())
            while i < len(events):
                yield i, events[i]
                i += 1
            if finished():
                return
