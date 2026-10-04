"""allow relations found through hyperlinks (method 'link')

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-04 20:35:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("relations_method_check", "relations", type_="check")
    op.create_check_constraint(
        "relations_method_check",
        "relations",
        "method IN ('regex', 'link', 'llm', 'annex', 'manual')",
    )


def downgrade() -> None:
    op.execute("DELETE FROM relations WHERE method = 'link'")
    op.drop_constraint("relations_method_check", "relations", type_="check")
    op.create_check_constraint(
        "relations_method_check",
        "relations",
        "method IN ('regex', 'llm', 'annex', 'manual')",
    )
