"""document full-text search column

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-04 18:30:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DOCUMENT_TSV = (
    "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
    "setweight(to_tsvector('english', coalesce(raw_text, '')), 'B')"
)


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(DOCUMENT_TSV, persisted=True),
            nullable=False,
        ),
    )
    op.create_index("ix_documents_tsv", "documents", ["tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_documents_tsv", table_name="documents")
    op.drop_column("documents", "tsv")
