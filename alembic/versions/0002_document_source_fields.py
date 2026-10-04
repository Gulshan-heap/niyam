"""document source fields

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-04 15:19:24.808697
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("source_id", sa.String(length=32), nullable=True))
    op.add_column("documents", sa.Column("rbi_no", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("updated_on", sa.Date(), nullable=True))
    op.add_column("documents", sa.Column("withdrawn_on", sa.Date(), nullable=True))
    op.add_column("documents", sa.Column("pdf_url", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("source_path", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True))
    op.create_unique_constraint(
        "documents_regulator_source_id_key", "documents", ["regulator", "source_id"]
    )


def downgrade() -> None:
    op.drop_constraint("documents_regulator_source_id_key", "documents", type_="unique")
    op.drop_column("documents", "fetched_at")
    op.drop_column("documents", "source_path")
    op.drop_column("documents", "pdf_url")
    op.drop_column("documents", "withdrawn_on")
    op.drop_column("documents", "updated_on")
    op.drop_column("documents", "rbi_no")
    op.drop_column("documents", "source_id")
