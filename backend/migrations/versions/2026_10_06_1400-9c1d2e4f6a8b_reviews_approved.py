"""reviews approved

Revision ID: 9c1d2e4f6a8b
Revises: 2f3e667a9c47
Create Date: 2026-10-06 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '9c1d2e4f6a8b'
down_revision: Union[str, Sequence[str], None] = '2f3e667a9c47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: no review so far has a decision. The new "waiting" status
    # needs no migration: status is a plain string column, not an enum.
    op.add_column('reviews', sa.Column('approved', postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('reviews', 'approved')
