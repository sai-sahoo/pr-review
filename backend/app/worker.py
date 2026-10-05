"""The worker: a separate process that runs reviews from the Redis queue.

The API only puts a job ("review <id>") into Redis and answers 202. This
process takes jobs out and runs the graph. Run as many workers as you like;
each job goes to exactly one of them.

Needs Redis and Postgres:  docker compose up -d  (from the repo root)

Run:  cd backend && arq app.worker.WorkerSettings
"""

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from datetime import UTC, datetime

from arq import ArqRedis
from arq.connections import RedisSettings
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.events import agent_event, node_event
from app.api.store import ReviewStore
from app.db import database_url
from app.github_client import GitHubError
from app.graph import build_graph
from app.graph.checkpointer import open_checkpointer

log = logging.getLogger(__name__)

REVIEW_TIMEOUT = 15 * 60  # seconds; a review still going after this is marked failed
MAX_CONCURRENT_REVIEWS = 4  # per worker process: caps parallel LLM calls (and cost)


def redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6380")  # docker-compose.yml's Redis


async def enqueue_review(redis: ArqRedis, review_id: str) -> None:
    """Put a review on the queue. Lives here, next to run_review, because the
    job names the function by its string name.

    _job_id=review_id makes this idempotent: while that job is queued or
    running, enqueuing it again does nothing. So a review can't run twice at once.
    """
    await redis.enqueue_job("run_review", review_id, _job_id=review_id)


async def startup(ctx: dict) -> None:
    """Runs once when the worker starts. Whatever goes into ctx is passed to every job.

    arq has already put its Redis connection in ctx["redis"]; the store
    reuses it to publish events.
    """
    db_url = ctx.get("db_url") or database_url()  # tests put a SQLite URL in ctx
    ctx["engine"] = engine = create_async_engine(db_url)
    ctx["store"] = store = ReviewStore(async_sessionmaker(engine, expire_on_commit=False), ctx["redis"])
    # open_checkpointer is an `async with` block, but startup and shutdown are
    # two separate functions. The exit stack keeps the block open in between.
    ctx["stack"] = stack = AsyncExitStack()
    ctx["graph"] = build_graph(await stack.enter_async_context(open_checkpointer(db_url)))

    # Postgres is the source of truth; the queue is a to-do list derived from
    # it. If Redis lost its data (it keeps it in memory), or an enqueue failed,
    # a review would be "queued" forever. Re-enqueue every unfinished one:
    # the ones still in the queue are skipped thanks to _job_id.
    for record in await store.unfinished():
        await enqueue_review(ctx["redis"], record.id)


async def shutdown(ctx: dict) -> None:
    await ctx["stack"].aclose()
    await ctx["engine"].dispose()


async def run_review(ctx: dict, review_id: str) -> None:
    """The job. arq calls it with ctx first, then the arguments given to enqueue_job.

    Only the id travels through Redis; everything else is read from Postgres,
    so the job always sees the current record.
    """
    store: ReviewStore = ctx["store"]
    graph = ctx["graph"]
    record = await store.get(review_id)
    # Queues deliver "at least once": the same job can arrive again, e.g. the
    # startup sweep after Redis lost track of a finished job. A finished
    # review must not be run (and paid for) twice.
    if record is None or record.status in ("done", "failed"):
        return

    # The checkpointer files this run's state under the thread_id. The
    # review id is already unique and stored, so it doubles as one.
    config = {"configurable": {"thread_id": review_id}}
    # A saved state means an earlier attempt got partway through this review
    # (the worker was stopped mid-job and arq handed the job out again).
    # Input None means "carry on from the last checkpoint"; a dict would
    # start a brand-new run.
    saved = await graph.aget_state(config)
    resuming = bool(saved.values)
    graph_input = None if resuming else {"pr_url": record.pr_url}

    await store.update(review_id, status="running")
    await store.add_event(review_id, {"type": "status", "status": "running", "resumed": resuming})
    try:
        # Our own deadline, raising TimeoutError, which the except below turns
        # into "failed". arq's job_timeout is only a backstop: when it fires,
        # arq cancels the job, and a cancelled job is left "running".
        async with asyncio.timeout(REVIEW_TIMEOUT):
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
        await fail(store, review_id, str(e))
        return
    except TimeoutError:
        await fail(store, review_id, f"timed out after {REVIEW_TIMEOUT // 60} minutes")
        return
    except Exception as e:
        # Caught here so the record says "failed" instead of staying
        # "running" forever. arq would only log it.
        log.exception("review %s crashed", review_id)
        await fail(store, review_id, f"internal error: {e}")
        return
    # Not caught: CancelledError (not an Exception). That's arq stopping the
    # job because the worker is shutting down. The record stays "running", arq
    # puts the job back in the queue, and the next worker resumes it from the
    # last checkpoint.

    # The full end state comes from the checkpoint. A resumed run only
    # streams the nodes it ran itself, so collecting the updates as they
    # arrive would miss what the earlier attempt did (fetch_pr's pr).
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


async def fail(store: ReviewStore, review_id: str, error: str) -> None:
    await store.update(review_id, status="failed", error=error, finished_at=datetime.now(UTC))
    await store.add_event(review_id, {"type": "failed", "error": error})


class WorkerSettings:
    """What `arq app.worker.WorkerSettings` reads. Each attribute is a Worker option."""

    functions = [run_review]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(redis_url())
    max_jobs = MAX_CONCURRENT_REVIEWS
    job_timeout = REVIEW_TIMEOUT + 60  # the backstop, after our own deadline
    # arq can store each job's return value in Redis. Our results live in
    # Postgres, so keep none (this also lets a finished id be enqueued again).
    keep_result = 0
