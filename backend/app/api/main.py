"""FastAPI wrapper: start a review, then poll for its result or stream its progress.

The API never runs a review itself. POST puts a job on the Redis queue, and a
worker process (app/worker.py) runs it. The API stays fast however many
reviews are in flight, and restarting it doesn't touch them.

    POST /reviews {"pr_url": ...}  -> 202 + a queued record (returns at once),
                                      or 200 + the existing review of that PR at that commit
    GET  /reviews/{id}             -> the record: queued -> running -> done | failed
    GET  /reviews                  -> all records, newest first
    GET  /reviews/{id}/events      -> live progress as Server-Sent Events
    POST /reviews/{id}/approval    -> a human's decision on a "waiting" review: which
       {"approved": [0, 2]}           findings to post. 202, and the worker carries on
    POST /webhooks/github          -> GitHub calls this when a PR is opened or pushed to;
                                      same dedupe, so a redelivery never starts a second review

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
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ValidationError
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.store import ReviewRecord, ReviewStore
from app.api.webhooks import PullRequestEvent, verify_signature
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


class ApprovalRequest(BaseModel):
    # Positions in the review's `findings` list. [] dismisses them all.
    approved: list[int]


def create_app(
    db_url: str | None = None,
    redis: ArqRedis | None = None,
    webhook_secret: str | None = None,
) -> FastAPI:
    """Build an app on the given database (default: DATABASE_URL), Redis
    (default: REDIS_URL) and webhook secret (default: GITHUB_WEBHOOK_SECRET).

    Tests pass a throwaway SQLite file and an in-memory fake Redis, so each
    test starts clean.
    """
    # The same string you type into the GitHub App's webhook settings. Empty
    # means webhooks are off: we never accept an unsigned delivery.
    webhook_secret = webhook_secret or os.getenv("GITHUB_WEBHOOK_SECRET", "")
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

    async def start_review(pr_url: str, head_sha: str) -> tuple[ReviewRecord, bool]:
        """(record, created). The live review of this PR at this commit, or a
        new one, already queued. Both entry points use it: the UI's POST and
        GitHub's webhook.
        """
        record, created = await store.get_or_create(pr_url, head_sha)
        if not created:
            return record, False
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
        return record, True

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

        record, created = await start_review(pr_url, head_sha)
        if not created:
            # Same PR, same commit: the code hasn't changed, so neither would
            # the review. Hand back the one we have (queued, running or done)
            # instead of paying for the LLM calls again. 200, not 202: no new work.
            response.status_code = 200
        return record

    @app.post("/webhooks/github")
    async def github_webhook(
        request: Request,
        response: Response,
        # FastAPI maps x_hub_signature_256 to the header X-Hub-Signature-256.
        # All optional, so a request without them reaches the signature check
        # and gets a 401, not a validation error.
        x_hub_signature_256: Annotated[str | None, Header()] = None,
        x_github_event: Annotated[str, Header()] = "",
        x_github_delivery: Annotated[str, Header()] = "",  # a unique id per delivery, for the logs
    ) -> dict:
        if not webhook_secret:
            raise HTTPException(status_code=503, detail="webhooks are off: set GITHUB_WEBHOOK_SECRET")
        # The raw bytes, exactly as sent. GitHub signed these bytes; JSON
        # parsed and dumped again could differ by a space and fail the check.
        body = await request.body()
        if not verify_signature(webhook_secret, body, x_hub_signature_256):
            log.warning("webhook %s: bad signature", x_github_delivery)
            raise HTTPException(status_code=401, detail="bad signature")

        # GitHub sends "ping" once, when the webhook is set up. Answering 2xx
        # shows a green tick in the App's Advanced -> Recent Deliveries.
        if x_github_event == "ping":
            return {"ok": "pong"}
        # Ignored events still get a 2xx: we received them fine, there's just
        # nothing to do. A 4xx/5xx would show up as a failed delivery.
        if x_github_event != "pull_request":
            return {"ignored": f"event {x_github_event!r}"}
        try:
            event = PullRequestEvent.model_validate_json(body)
        except ValidationError:  # signed by GitHub, but not the shape we expect
            raise HTTPException(status_code=422, detail="unexpected pull_request payload")
        if reason := event.skip_reason():
            return {"ignored": reason}

        # No GitHub API call needed: the payload already has the canonical URL
        # and the head commit. GitHub waits at most 10 s for our answer, and
        # we only insert a row and queue a job, so we answer in milliseconds.
        pr = event.pull_request
        record, created = await start_review(pr.html_url, pr.head.sha)
        log.info("webhook %s: %s %s -> review %s (%s)", x_github_delivery, event.action,
                 pr.html_url, record.id, "new" if created else "existing")
        response.status_code = 202 if created else 200
        return {"review_id": record.id, "created": created}

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

    @app.post("/reviews/{review_id}/approval", status_code=202, response_model=ReviewRecord)
    async def approve_review(
        body: ApprovalRequest,
        record: Annotated[ReviewRecord, Depends(existing_review)],
    ) -> ReviewRecord:
        # 409 Conflict: the request is fine, but not in the review's current state.
        if record.status != "waiting":
            raise HTTPException(status_code=409, detail=f"review is {record.status}, not waiting for approval")
        if any(not 0 <= i < len(record.findings) for i in body.approved):
            raise HTTPException(status_code=422, detail=f"approved must be positions 0-{len(record.findings) - 1}")
        approved = sorted(set(body.approved))  # a double click's [0, 0] is just [0]

        # The status check above was a friendly early answer. This is the
        # real one: of two racing requests, only one changes the row.
        if not await store.approve(record.id, approved):
            raise HTTPException(status_code=409, detail="review was already decided")
        try:
            await enqueue_review(redis, record.id)
        except RedisError:
            # The decision is saved but no job will act on it. Undo, so the
            # human sees "waiting" again and can resend, instead of a review
            # stuck in "queued" until a worker restarts.
            log.exception("could not queue approved review %s", record.id)
            await store.update(record.id, status="waiting", approved=None)
            raise HTTPException(status_code=503, detail="review queue unavailable, try again shortly")
        return await store.get(record.id)

    @app.get("/reviews", response_model=list[ReviewRecord])
    async def list_reviews() -> list[ReviewRecord]:
        return await store.list_all()

    return app


app = create_app()  # what `uvicorn app.api.main:app` loads
