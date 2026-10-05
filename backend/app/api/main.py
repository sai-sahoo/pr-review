"""FastAPI wrapper: start a review, then poll for its result or stream its progress.

    POST /reviews {"pr_url": ...}  -> 202 + a queued record (returns at once)
    GET  /reviews/{id}             -> the record: queued -> running -> done | failed
    GET  /reviews                  -> all records, newest first
    GET  /reviews/{id}/events      -> live progress as Server-Sent Events

Needs Postgres:  docker compose up -d  (from the repo root)
                 cd backend && alembic upgrade head

Run:  cd backend && uv run uvicorn app.api.main:app --reload
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

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.events import agent_event, node_event
from app.api.store import ReviewRecord, ReviewStore
from app.db import database_url
from app.github_client import GitHubError, parse_pr_url
from app.graph import build_graph
from app.graph.checkpointer import open_checkpointer

log = logging.getLogger(__name__)

# The Next.js UI runs on another port, which makes it another "origin". Browsers
# block a page from reading responses from a different origin unless the server
# says that origin is allowed. Comma-separated, so you can list several.
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")


class ReviewRequest(BaseModel):
    pr_url: str


def create_app(db_url: str | None = None) -> FastAPI:
    """Build an app on the given database (default: DATABASE_URL).

    Tests pass a throwaway SQLite file, so each test starts clean.
    """
    # The engine opens connections lazily and keeps them in a pool for reuse.
    # Creating it here doesn't connect yet; the first query does.
    db_url = db_url or database_url()
    engine = create_async_engine(db_url)
    # expire_on_commit=False: after a commit, keep the row's values in Python
    # instead of reloading them from the database on the next attribute read.
    store = ReviewStore(async_sessionmaker(engine, expire_on_commit=False))

    # The reviews running right now, as asyncio tasks. The event loop only
    # keeps a weak reference to a task, so without this set a running review
    # could be garbage-collected halfway through.
    jobs: set[asyncio.Task] = set()

    def start_job(record: ReviewRecord) -> None:
        task = asyncio.create_task(run_review(record.id, record.pr_url))
        jobs.add(task)
        task.add_done_callback(jobs.discard)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Before `yield`: startup, before the server accepts any request.
        async with open_checkpointer(db_url) as checkpointer:
            app.state.graph = build_graph(checkpointer)  # compiled once, used by every review
            # A review still queued or running lost its task when the last
            # server process stopped. Its checkpoints are saved, so pick it up
            # again. In the background: startup must not wait minutes for it.
            for record in await store.unfinished():
                start_job(record)

            yield  # the server runs here

            # Shutdown (Ctrl+C, or --reload after you save a file). Stop the
            # running reviews now instead of waiting for them. They stay
            # "running" in the database, so the next startup resumes them.
            for task in jobs:
                task.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
        await engine.dispose()  # close the pooled connections

    app = FastAPI(title="pr-review", lifespan=lifespan)
    app.state.store = store  # lets tests reach the store,
    app.state.jobs = jobs  # and wait for the reviews to finish
    # Adds the Access-Control-Allow-* headers, and answers the browser's
    # OPTIONS "preflight" question before a JSON POST.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=FRONTEND_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    async def run_review(review_id: str, pr_url: str) -> None:
        """The slow part, run after POST has answered (or resumed at startup).

        It's `async def`, so it runs on the event loop. That's safe because
        astream() never blocks the loop: LangGraph runs our plain-`def` nodes in
        worker threads and awaits them.
        """
        graph = app.state.graph
        # The checkpointer files this run's state under the thread_id. The
        # review id is already unique and stored, so it doubles as one.
        config = {"configurable": {"thread_id": review_id}}
        # A saved state means an earlier process got partway through this
        # review. Input None means "carry on from the last checkpoint"; a dict
        # would start a brand-new run.
        saved = await graph.aget_state(config)
        resuming = bool(saved.values)
        graph_input = None if resuming else {"pr_url": pr_url}

        await store.update(review_id, status="running")
        await store.add_event(review_id, {"type": "status", "status": "running", "resumed": resuming})
        try:
            # Two stream modes at once, so each chunk is a (mode, data) pair:
            #   "updates": {node: update} each time a node finishes
            #   "custom":  whatever a node passed to get_stream_writer(), while it runs
            async for mode, chunk in graph.astream(graph_input, config, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    await store.add_event(review_id, agent_event(chunk))
                    continue
                for node, update in chunk.items():
                    await store.add_event(review_id, node_event(node, update))
        except GitHubError as e:  # a problem the user can fix: show it as is
            await fail(review_id, str(e))
            return
        except Exception as e:
            # A task has no caller to raise to. Without this the
            # error is only logged and the record stays "running" forever.
            log.exception("review %s crashed", review_id)
            await fail(review_id, f"internal error: {e}")
            return
        # The full end state comes from the checkpoint. A resumed run only
        # streams the nodes it ran itself, so collecting the updates as they
        # arrive would miss what the earlier process did (fetch_pr's pr).
        final = (await graph.aget_state(config)).values
        await store.update(
            review_id,
            status="done",
            title=final["pr"].title,
            findings=final["verified"],
            checks=final["checks"],
            finished_at=datetime.now(UTC),
        )
        # The record is final before this event goes out, so a client that
        # GETs the record after seeing "done" always sees the findings.
        await store.add_event(review_id, {"type": "done", "kept": len(final["verified"])})

    async def fail(review_id: str, error: str) -> None:
        await store.update(review_id, status="failed", error=error, finished_at=datetime.now(UTC))
        await store.add_event(review_id, {"type": "failed", "error": error})

    @app.post("/reviews", status_code=202, response_model=ReviewRecord)
    async def create_review(body: ReviewRequest) -> ReviewRecord:
        # Reject a bad URL now, while the caller is still waiting for an answer.
        try:
            parse_pr_url(body.pr_url)
        except GitHubError as e:
            raise HTTPException(status_code=422, detail=str(e))
        record = await store.create(body.pr_url.strip())
        # A task on the event loop, not BackgroundTasks: shutdown can cancel
        # it (see lifespan), and the caller gets the id right away.
        start_job(record)
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
