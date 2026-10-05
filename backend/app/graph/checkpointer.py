"""Where LangGraph saves the graph's state after every step (a "checkpoint").

Compile a graph with a checkpointer and give each run a thread_id. After every
super-step, LangGraph writes the whole state under that id, plus which nodes
are due next. If the process dies, calling the graph again with the same
thread_id and input None carries on from the last saved step: finished nodes
(and their LLM calls) are not repeated.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy.engine import make_url

from app.schemas import Finding, FindingCheck, PullRequest, TriagePlan

# Checkpoints store our Pydantic objects as (module, class name, data), and
# loading one imports that class. Only these may be loaded, so a tampered
# checkpoint row can't make us import and build anything else.
SERDE = JsonPlusSerializer(allowed_msgpack_modules=[PullRequest, TriagePlan, Finding, FindingCheck])


@asynccontextmanager
async def open_checkpointer(db_url: str) -> AsyncIterator[BaseCheckpointSaver]:
    """A checkpointer on the same database as the app, open until the block ends."""
    url = make_url(db_url)
    if url.get_backend_name() == "postgresql":
        # psycopg itself wants a plain postgresql:// URL, without SQLAlchemy's "+psycopg".
        conninfo = url.set(drivername="postgresql").render_as_string(hide_password=False)
        async with AsyncPostgresSaver.from_conn_string(conninfo, serde=SERDE) as saver:
            # Creates the checkpoint tables if missing. They have their own
            # built-in migrations, so they stay out of our Alembic history.
            await saver.setup()
            yield saver
    else:
        # Tests run on SQLite. Imported here because it's a dev-only dependency.
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with aiosqlite.connect(url.database or ":memory:") as conn:
            saver = AsyncSqliteSaver(conn, serde=SERDE)
            await saver.setup()
            yield saver
