"""chunk context and longer section_ref

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-04 18:50:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CHUNK_TSV = (
    "setweight(to_tsvector('english', coalesce(context, '')), 'A') || "
    "setweight(to_tsvector('english', text), 'B')"
)


def upgrade() -> None:
    op.alter_column("chunks", "section_ref", type_=sa.Text(), existing_nullable=True)
    op.add_column("chunks", sa.Column("context", sa.Text(), nullable=True))
    # A generated column's expression can't be altered in place: rebuild it.
    op.drop_index("ix_chunks_tsv", table_name="chunks")
    op.drop_column("chunks", "tsv")
    op.add_column(
        "chunks",
        sa.Column(
            "tsv", postgresql.TSVECTOR(), sa.Computed(CHUNK_TSV, persisted=True), nullable=False
        ),
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_tsv", table_name="chunks")
    op.drop_column("chunks", "tsv")
    op.add_column(
        "chunks",
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=False,
        ),
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.drop_column("chunks", "context")
    op.alter_column("chunks", "section_ref", type_=sa.String(length=64), existing_nullable=True)
