"""Reviews and their events, stored in Postgres.

Same methods as the Step 9 dict store, now `async`: each one waits for the
database, and while it waits the event loop serves other requests.

One thing stays in memory: a per-review "new event!" signal that wakes up the
SSE handlers. It only reaches handlers in this same process. Step 13 swaps it
for Redis pub/sub, so several processes can share it.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.events import TERMINAL
from app.db import EventRow, ReviewRow
from app.schemas import Finding, FindingCheck

Status = Literal["queued", "running", "done", "failed"]
UNFINISHED = ["queued", "running"]


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


# The API speaks ReviewRecord (Pydantic); the database speaks ReviewRow
# (SQLAlchemy). These two convert, so neither side needs to know the other.

def _to_record(row: ReviewRow) -> ReviewRecord:
    # from_attributes: read row.id, row.status… instead of row["id"]. The
    # findings come back as plain dicts and are validated into Finding objects.
    return ReviewRecord.model_validate(row, from_attributes=True)


def _to_columns(record: ReviewRecord) -> dict[str, Any]:
    columns = record.model_dump()
    # The JSON columns need JSON-safe values (enums -> strings), while the
    # datetime columns want real datetimes, so only these two use mode="json".
    columns |= record.model_dump(mode="json", include={"findings", "checks"})
    return columns


class ReviewStore:
    def __init__(self, sessions: async_sessionmaker) -> None:
        # Each `async with self._sessions()` borrows a connection from the
        # pool and gives it back at the end of the block.
        self._sessions = sessions

        # In-memory wake-up signal, per review: the seq of its newest event,
        # and a Condition the SSE handlers sleep on until that number grows.
        self._last_seq: dict[str, int] = {}
        self._new_event: dict[str, asyncio.Condition] = {}

    async def create(self, pr_url: str) -> ReviewRecord:
        record = ReviewRecord(pr_url=pr_url)
        # .begin(): COMMIT at the end of the block, or ROLLBACK if it raises.
        async with self._sessions.begin() as session:
            session.add(ReviewRow(**_to_columns(record)))
        return record

    async def get(self, review_id: str) -> ReviewRecord | None:
        async with self._sessions() as session:
            row = await session.get(ReviewRow, review_id)  # SELECT … WHERE id = :id
        return _to_record(row) if row else None

    async def list_all(self) -> list[ReviewRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(select(ReviewRow).order_by(ReviewRow.created_at.desc()))
            return [_to_record(row) for row in rows]

    async def unfinished(self) -> list[ReviewRecord]:
        """Reviews still queued or running according to the database."""
        async with self._sessions() as session:
            rows = await session.scalars(select(ReviewRow).where(ReviewRow.status.in_(UNFINISHED)))
            return [_to_record(row) for row in rows]

    async def update(self, review_id: str, **changes) -> ReviewRecord:
        async with self._sessions.begin() as session:
            row = await session.get(ReviewRow, review_id)
            record = _to_record(row).model_copy(update=changes)
            # Setting attributes marks the row "dirty"; the commit at the end
            # of the block sends one UPDATE with the changed columns.
            for column, value in _to_columns(record).items():
                setattr(row, column, value)
        return record

    # --- events ---------------------------------------------------------
    # Only ever called from the event loop: asyncio.Condition is not thread-safe.

    def _condition(self, review_id: str) -> asyncio.Condition:
        return self._new_event.setdefault(review_id, asyncio.Condition())

    async def add_event(self, review_id: str, event: dict[str, Any]) -> None:
        async with self._sessions.begin() as session:
            # Next number = highest so far + 1. Only one task writes a review's
            # events, and the primary key would reject a duplicate anyway.
            last = await session.scalar(select(func.max(EventRow.seq)).where(EventRow.review_id == review_id))
            seq = 0 if last is None else last + 1
            session.add(EventRow(review_id=review_id, seq=seq, type=event["type"], data=event))
        # Committed, so any reader that wakes up now will find the row.
        self._last_seq[review_id] = seq
        condition = self._condition(review_id)
        async with condition:  # notify_all() requires holding the condition's lock
            condition.notify_all()

    async def follow(self, review_id: str, start: int = 0) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        """Yield (seq, event) from `start` on: first the stored ones, then live
        ones as they arrive. Ends after the done/failed event.
        """
        i = start
        while True:
            async with self._sessions() as session:
                rows = (await session.scalars(
                    select(EventRow)
                    .where(EventRow.review_id == review_id, EventRow.seq >= i)
                    .order_by(EventRow.seq)
                )).all()
                if not rows:
                    # Nothing new. Already past the final event (a resume after
                    # done/failed)? Then end now instead of waiting forever.
                    newest = await session.scalar(
                        select(EventRow.type).where(EventRow.review_id == review_id)
                        .order_by(EventRow.seq.desc()).limit(1)
                    )
                    if newest in TERMINAL:
                        return
            # The connection is back in the pool before we yield: a slow client
            # must not hold one for minutes.
            for row in rows:
                yield row.seq, row.data
                if row.type in TERMINAL:
                    return
                i = row.seq + 1

            condition = self._condition(review_id)
            async with condition:
                # Sleep until add_event stores an event at position i or later.
                # wait_for checks first, so an event added between our SELECT
                # and this line is not missed.
                await condition.wait_for(lambda: self._last_seq.get(review_id, -1) >= i)
