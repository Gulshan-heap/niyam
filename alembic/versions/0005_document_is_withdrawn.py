"""document is_withdrawn

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-04 18:35:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("is_withdrawn", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.execute("UPDATE documents SET is_withdrawn = true WHERE withdrawn_on IS NOT NULL")
    # RBI reuses one watermark image, so its date can precede the document itself.
    op.execute(
        "UPDATE documents SET withdrawn_on = NULL, valid_to = NULL WHERE withdrawn_on < issued_date"
    )


def downgrade() -> None:
    op.drop_column("documents", "is_withdrawn")
