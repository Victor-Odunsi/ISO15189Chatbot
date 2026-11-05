"""add session_summaries table (bounded chat history)

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "session_summaries",
        sa.Column("session_id", sa.String(length=64), primary_key=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("summarized_up_to_id", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("session_summaries")
