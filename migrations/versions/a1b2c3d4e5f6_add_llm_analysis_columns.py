"""Add LLM analysis columns.

Revision ID: a1b2c3d4e5f6
Revises: 7d7aae5b6d1a
Create Date: 2026-02-09 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "7d7aae5b6d1a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("articles", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column("articles", sa.Column("key_points", sa.Text(), nullable=True))
    op.add_column("articles", sa.Column("implication", sa.Text(), nullable=True))
    op.add_column("articles", sa.Column("category", sa.String(128), nullable=True))


def downgrade() -> None:
    op.drop_column("articles", "category")
    op.drop_column("articles", "implication")
    op.drop_column("articles", "key_points")
    op.drop_column("articles", "summary")
