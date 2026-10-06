"""Reviews and their events, stored in Postgres.

Same methods as the Step 9 dict store, now `async`: each one waits for the
database, and while it waits the event loop serves other requests.

Reviews now run in a worker process, while the SSE handlers live in the API
process. So the "new event!" signal goes through Redis pub/sub: add_event
publishes on the review's channel, follow subscribes to it. The message is
only a wake-up call; the events themselves are always read from Postgres.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.events import TERMINAL
from app.db import EventRow, ReviewRow
from app.schemas import Finding, FindingCheck

Status = Literal["queued", "running", "done", "failed"]
UNFINISHED = ["queued", "running"]

# Pub/sub is fire-and-forget: a message sent while nobody listens, or lost in
# a Redis reconnect, is gone. So a waiting reader also re-checks the database
# this often, even without a message. A missed message costs a delay, never a hang.
POLL_SECONDS = 15


def _now() -> datetime:
    return datetime.now(UTC)


class ReviewRecord(BaseModel):
    """One review job, as the API returns it."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    pr_url: str
    head_sha: str | None = None  # the commit under review (None: from before Step 14a)
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


def _channel(review_id: str) -> str:
    return f"review:{review_id}:events"


class ReviewStore:
    def __init__(self, sessions: async_sessionmaker, redis: Redis) -> None:
        # Each `async with self._sessions()` borrows a connection from the
        # pool and gives it back at the end of the block.
        self._sessions = sessions
        self._redis = redis

    async def create(self, pr_url: str, head_sha: str | None = None) -> ReviewRecord:
        record = ReviewRecord(pr_url=pr_url, head_sha=head_sha)
        # .begin(): COMMIT at the end of the block, or ROLLBACK if it raises.
        async with self._sessions.begin() as session:
            session.add(ReviewRow(**_to_columns(record)))
        return record

    async def find_live(self, pr_url: str, head_sha: str) -> ReviewRecord | None:
        """The queued, running or done review of this PR at this commit, if any."""
        async with self._sessions() as session:
            row = await session.scalar(
                select(ReviewRow).where(
                    ReviewRow.pr_url == pr_url,
                    ReviewRow.head_sha == head_sha,
                    ReviewRow.status != "failed",  # the same filter as the partial index
                )
            )
        return _to_record(row) if row else None

    async def get_or_create(self, pr_url: str, head_sha: str) -> tuple[ReviewRecord, bool]:
        """(record, created). The existing live review of this PR+commit, or a new one.

        The SELECT is the fast path. The unique index is the real guard: if
        another request inserts the same key between our SELECT and INSERT,
        our INSERT fails with IntegrityError, and we return the winner's row.
        """
        if existing := await self.find_live(pr_url, head_sha):
            return existing, False
        try:
            return await self.create(pr_url, head_sha), True
        except IntegrityError:
            existing = await self.find_live(pr_url, head_sha)
            if existing is None:  # the winner failed in those few ms; rare, let the caller retry
                raise
            return existing, False

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

    async def add_event(self, review_id: str, event: dict[str, Any]) -> None:
        async with self._sessions.begin() as session:
            # Next number = highest so far + 1. Only one task writes a review's
            # events, and the primary key would reject a duplicate anyway.
            last = await session.scalar(select(func.max(EventRow.seq)).where(EventRow.review_id == review_id))
            seq = 0 if last is None else last + 1
            session.add(EventRow(review_id=review_id, seq=seq, type=event["type"], data=event))
        # Committed, so any reader that wakes up now will find the row. Every
        # process subscribed to this channel gets the message (the seq is just
        # for logs and redis-cli; readers don't need it).
        await self._redis.publish(_channel(review_id), seq)

    async def follow(self, review_id: str, start: int = 0) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        """Yield (seq, event) from `start` on: first the stored ones, then live
        ones as they arrive. Ends after the done/failed event.
        """
        async with self._redis.pubsub() as pubsub:
            # Subscribe *before* the first SELECT. An event stored after this
            # line sends us a message; one stored before it is in the SELECT.
            # Subscribing after the SELECT would leave a gap where an event
            # could slip through unseen.
            await pubsub.subscribe(_channel(review_id))
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

                # Sleep until a message arrives (or POLL_SECONDS pass), then
                # SELECT again. The first call returns at once with None: that
                # was Redis confirming the subscription, which we skip.
                await pubsub.get_message(ignore_subscribe_messages=True, timeout=POLL_SECONDS)
