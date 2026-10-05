"""Database tables, described as Python classes (SQLAlchemy's ORM).

Two tables:
    reviews        one row per review job (what GET /reviews/{id} returns)
    review_events  every SSE event of every review, in order (what the UI replays)

These classes describe what the tables *should* look like. Alembic compares
them with the real database and writes the migration that gets it there.
"""

import os
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on Postgres (binary, indexable); plain JSON elsewhere (SQLite in tests).
Json = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Every table class inherits this, so Base.metadata knows all tables."""


class ReviewRow(Base):
    __tablename__ = "reviews"

    # Mapped[str] -> NOT NULL; Mapped[str | None] -> NULL allowed
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    pr_url: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), index=True)  # startup looks up unfinished ones
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    title: Mapped[str | None] = mapped_column(Text)
    # Lists of Finding / FindingCheck, stored as JSON. We never query inside
    # them, so separate tables would be extra joins for nothing.
    findings: Mapped[list[dict[str, Any]]] = mapped_column(Json, default=list)
    checks: Mapped[list[dict[str, Any]]] = mapped_column(Json, default=list)
    error: Mapped[str | None] = mapped_column(Text)


class EventRow(Base):
    __tablename__ = "review_events"

    # (review_id, seq) together are the key: event 0, 1, 2… of one review.
    # seq is also the SSE "id:", so Last-Event-ID resumes after it.
    review_id: Mapped[str] = mapped_column(ForeignKey("reviews.id", ondelete="CASCADE"), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(32))
    data: Mapped[dict[str, Any]] = mapped_column(Json)


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set; see .env.example")
    return url

