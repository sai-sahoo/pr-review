"""reviews head_sha and live pr+sha unique index

Revision ID: 2f3e667a9c47
Revises: 4bf7b19ed8af
Create Date: 2026-10-05 20:17:41.276139

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2f3e667a9c47'
down_revision: Union[str, Sequence[str], None] = '4bf7b19ed8af'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: existing rows don't know their commit and stay NULL, which
    # the unique index ignores (NULLs never count as equal).
    op.add_column('reviews', sa.Column('head_sha', sa.String(length=40), nullable=True))
    # Written by hand: autogenerate doesn't reliably pick up partial-index WHERE clauses.
    op.create_index(
        'uq_reviews_live_pr_sha', 'reviews', ['pr_url', 'head_sha'], unique=True,
        postgresql_where=sa.text("status <> 'failed'"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('uq_reviews_live_pr_sha', table_name='reviews')
    op.drop_column('reviews', 'head_sha')
