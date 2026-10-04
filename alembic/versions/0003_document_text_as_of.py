"""document text_as_of

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-04 17:59:44.961368
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("text_as_of", sa.Date(), nullable=True))
    # Existing rows hold the text of the page as published.
    op.execute("UPDATE documents SET text_as_of = COALESCE(updated_on, issued_date)")


def downgrade() -> None:
    op.drop_column("documents", "text_as_of")
