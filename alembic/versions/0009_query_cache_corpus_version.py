"""query_cache.corpus_version

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-04 21:15:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("query_cache", sa.Column("corpus_version", sa.String(length=64), nullable=True))
    op.create_index("ix_query_cache_corpus_version", "query_cache", ["corpus_version"])


def downgrade() -> None:
    op.drop_index("ix_query_cache_corpus_version", table_name="query_cache")
    op.drop_column("query_cache", "corpus_version")
