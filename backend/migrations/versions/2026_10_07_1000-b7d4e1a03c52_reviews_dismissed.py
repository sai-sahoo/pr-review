"""reviews dismissed

Revision ID: b7d4e1a03c52
Revises: 9c1d2e4f6a8b
Create Date: 2026-10-07 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b7d4e1a03c52'
down_revision: Union[str, Sequence[str], None] = '9c1d2e4f6a8b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # NOT NULL with a server default: Postgres fills '[]' into every existing
    # row as it adds the column, so old reviews read back as "nothing dismissed".
    op.add_column('reviews', sa.Column('dismissed', postgresql.JSONB(),
                                       server_default=sa.text("'[]'"), nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('reviews', 'dismissed')
