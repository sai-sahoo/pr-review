"""FastAPI wrapper: start a review, then poll for its result or stream its progress.

The API never runs a review itself. POST puts a job on the Redis queue, and a
worker process (app/worker.py) runs it. The API stays fast however many
reviews are in flight, and restarting it doesn't touch them.

    POST /reviews {"pr_url": ...}  -> 202 + a queued record (returns at once),
                                      or 200 + the existing review of that PR at that commit
    GET  /reviews/{id}             -> the record: queued -> running -> done | failed
    GET  /reviews                  -> all records, newest first
    GET  /reviews/{id}/events      -> live progress as Server-Sent Events

Needs Postgres and Redis:  docker compose up -d  (from the repo root)
                           cd backend && alembic upgrade head

Run:  cd backend && uv run uvicorn app.api.main:app --reload
      and, in a second terminal, the worker:  arq app.worker.WorkerSettings
      then open http://127.0.0.1:8000/docs for an interactive page
      watch a review live:  curl -N localhost:8000/reviews/<id>/events
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

import httpx
from arq import ArqRedis
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.store import ReviewRecord, ReviewStore
from app.db import database_url
from app.github_client import GitHubError, get_pr_head, parse_pr_url
from app.worker import enqueue_review, redis_url

log = logging.getLogger(__name__)

# The Next.js UI runs on another port, which makes it another "origin". Browsers
# block a page from reading responses from a different origin unless the server
# says that origin is allowed. Comma-separated, so you can list several.
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")


class ReviewRequest(BaseModel):
    pr_url: str


def create_app(db_url: str | None = None, redis: ArqRedis | None = None) -> FastAPI:
    """Build an app on the given database (default: DATABASE_URL) and Redis
    (default: REDIS_URL).

    Tests pass a throwaway SQLite file and an in-memory fake Redis, so each
    test starts clean.
    """
    # The engine opens connections lazily and keeps them in a pool for reuse.
    # Creating it here doesn't connect yet; the first query does.
    db_url = db_url or database_url()
    engine = create_async_engine(db_url)
    # Same for Redis: a connection pool that connects on first use. ArqRedis
    # is the regular redis-py client plus arq's enqueue_job.
    redis = redis or ArqRedis.from_url(redis_url())
    # expire_on_commit=False: after a commit, keep the row's values in Python
    # instead of reloading them from the database on the next attribute read.
    store = ReviewStore(async_sessionmaker(engine, expire_on_commit=False), redis)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Nothing to start or resume any more: unfinished reviews are the
        # worker's business. Only close the connection pools on shutdown.
        yield
        await redis.aclose()
        await engine.dispose()

    app = FastAPI(title="pr-review", lifespan=lifespan)
    app.state.store = store  # lets tests reach the store,
    app.state.redis = redis  # the queue,
    app.state.db_url = db_url  # and give a test worker the same database
    # Adds the Access-Control-Allow-* headers, and answers the browser's
    # OPTIONS "preflight" question before a JSON POST.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=FRONTEND_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    @app.post("/reviews", status_code=202, response_model=ReviewRecord)
    async def create_review(body: ReviewRequest, response: Response) -> ReviewRecord:
        # Reject a bad URL now, while the caller is still waiting for an answer.
        try:
            parse_pr_url(body.pr_url)
            # The dedupe key needs the PR's current head commit: one small
            # GitHub request. httpx here is the sync client, so run it in a
            # thread; called directly, it would freeze the event loop (every
            # other request, every SSE stream) while GitHub answers.
            pr_url, head_sha = await asyncio.to_thread(get_pr_head, body.pr_url)
        except GitHubError as e:  # bad URL, PR not found, rate limit: the user can act on it
            raise HTTPException(status_code=422, detail=str(e))
        except httpx.HTTPError:
            log.exception("GitHub lookup failed for %s", body.pr_url)
            raise HTTPException(status_code=502, detail="could not reach GitHub, try again shortly")

        record, created = await store.get_or_create(pr_url, head_sha)
        if not created:
            # Same PR, same commit: the code hasn't changed, so neither would
            # the review. Hand back the one we have (queued, running or done)
            # instead of paying for the LLM calls again. 200, not 202: no new work.
            response.status_code = 200
            return record

        # Order matters: the row is committed before the job exists, so a
        # worker that picks the job up a millisecond later finds the record.
        try:
            await enqueue_review(redis, record.id)
        except RedisError:
            # Two systems, no shared transaction: the row is saved but the job
            # isn't. Say so, instead of leaving a review "queued" forever.
            # "failed" also takes it out of the unique index, so a retry can
            # create a fresh review for this commit.
            log.exception("could not queue review %s", record.id)
            await store.update(record.id, status="failed", error="queue unavailable",
                               finished_at=datetime.now(UTC))
            raise HTTPException(status_code=503, detail="review queue unavailable, try again shortly")
        return record

    async def existing_review(review_id: str) -> ReviewRecord:
        """Dependency: the record for the path's review_id, or a 404."""
        record = await store.get(review_id)
        if record is None:
            raise HTTPException(status_code=404, detail="review not found")
        return record

    @app.get("/reviews/{review_id}", response_model=ReviewRecord)
    async def get_review(record: Annotated[ReviewRecord, Depends(existing_review)]) -> ReviewRecord:
        return record

    @app.get("/reviews/{review_id}/events", response_class=EventSourceResponse)
    async def review_events(
        # A dependency runs before the response starts. A check inside this
        # generator would be too late: by its first line, 200 is already sent.
        record: Annotated[ReviewRecord, Depends(existing_review)],
        # On reconnect, the browser's EventSource sends back the id of the last
        # event it received, so we resume after it instead of starting over.
        last_event_id: Annotated[int | None, Header()] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        start = 0 if last_event_id is None else last_event_id + 1
        # Each event goes out as "id: <index>", "event: <type>", "data: <json>".
        # FastAPI also sends a ": ping" comment every 15s of silence, so
        # proxies don't close a connection that's quiet while an LLM thinks.
        async for i, event in store.follow(record.id, start):
            yield ServerSentEvent(id=str(i), event=event["type"], data=event)

    @app.get("/reviews", response_model=list[ReviewRecord])
    async def list_reviews() -> list[ReviewRecord]:
        return await store.list_all()

    return app


app = create_app()  # what `uvicorn app.api.main:app` loads
