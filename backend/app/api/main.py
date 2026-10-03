"""FastAPI wrapper: start a review, then poll for its result.

    POST /reviews {"pr_url": ...}  -> 202 + a queued record (returns at once)
    GET  /reviews/{id}             -> the record: queued -> running -> done | failed
    GET  /reviews                  -> all records, newest first

Run:  cd backend && uv run uvicorn app.api.main:app --reload
      then open http://127.0.0.1:8000/docs for an interactive page
"""

import logging
from datetime import UTC, datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from app.api.store import ReviewRecord, ReviewStore
from app.github_client import GitHubError, parse_pr_url
from app.graph import build_graph

log = logging.getLogger(__name__)


class ReviewRequest(BaseModel):
    pr_url: str


def create_app() -> FastAPI:
    """Build a fresh app with its own store. Tests call this to start clean."""
    app = FastAPI(title="pr-review")
    store = ReviewStore()
    graph = build_graph()  # compiled once, reused by every request

    def run_review(review_id: str, pr_url: str) -> None:
        """The slow part. A plain `def`, so FastAPI runs it in a worker thread."""
        store.update(review_id, status="running")
        try:
            state = graph.invoke({"pr_url": pr_url})
        except GitHubError as e:  # a problem the user can fix: show it as is
            store.update(review_id, status="failed", error=str(e), finished_at=datetime.now(UTC))
            return
        except Exception as e:
            # A background task has no caller to raise to. Without this the
            # error is only logged and the record stays "running" forever.
            log.exception("review %s crashed", review_id)
            store.update(review_id, status="failed", error=f"internal error: {e}", finished_at=datetime.now(UTC))
            return
        store.update(
            review_id,
            status="done",
            title=state["pr"].title,
            findings=state["verified"],
            checks=state["checks"],
            finished_at=datetime.now(UTC),
        )

    @app.post("/reviews", status_code=202, response_model=ReviewRecord)
    async def create_review(body: ReviewRequest, background: BackgroundTasks) -> ReviewRecord:
        # Reject a bad URL now, while the caller is still waiting for an answer.
        try:
            parse_pr_url(body.pr_url)
        except GitHubError as e:
            raise HTTPException(status_code=422, detail=str(e))
        record = store.create(body.pr_url.strip())
        # Runs after the response is sent, so the caller gets the id right away.
        background.add_task(run_review, record.id, record.pr_url)
        return record

    @app.get("/reviews/{review_id}", response_model=ReviewRecord)
    async def get_review(review_id: str) -> ReviewRecord:
        record = store.get(review_id)
        if record is None:
            raise HTTPException(status_code=404, detail="review not found")
        return record

    @app.get("/reviews", response_model=list[ReviewRecord])
    async def list_reviews() -> list[ReviewRecord]:
        return store.list()

    return app


app = create_app()  # what `uvicorn app.api.main:app` loads
