"""key alerts on the new document instead of the relation

Relations are rebuilt daily with new ids, so alerts keyed on relation ids would repeat.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-04 21:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DELETE FROM alerts_sent")  # no alerts have been sent yet
    op.drop_constraint("alerts_sent_subscription_id_relation_id_key", "alerts_sent", type_="unique")
    op.drop_constraint("alerts_sent_relation_id_fkey", "alerts_sent", type_="foreignkey")
    op.drop_column("alerts_sent", "relation_id")
    op.add_column("alerts_sent", sa.Column("document_id", sa.Integer(), nullable=False))
    op.create_foreign_key(
        "alerts_sent_document_id_fkey",
        "alerts_sent",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "alerts_sent_subscription_id_document_id_key",
        "alerts_sent",
        ["subscription_id", "document_id"],
    )


def downgrade() -> None:
    op.execute("DELETE FROM alerts_sent")
    op.drop_constraint("alerts_sent_subscription_id_document_id_key", "alerts_sent", type_="unique")
    op.drop_constraint("alerts_sent_document_id_fkey", "alerts_sent", type_="foreignkey")
    op.drop_column("alerts_sent", "document_id")
    op.add_column("alerts_sent", sa.Column("relation_id", sa.Integer(), nullable=False))
    op.create_foreign_key(
        "alerts_sent_relation_id_fkey",
        "alerts_sent",
        "relations",
        ["relation_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "alerts_sent_subscription_id_relation_id_key",
        "alerts_sent",
        ["subscription_id", "relation_id"],
    )
