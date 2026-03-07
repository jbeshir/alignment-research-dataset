"""add thumbnail_url

Revision ID: b3c4d5e6f7a8
Revises: a1b2c3d4e5f6
Create Date: 2026-03-04

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b3c4d5e6f7a8"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("articles", sa.Column("thumbnail_url", sa.String(2048), nullable=True))


def downgrade() -> None:
    op.drop_column("articles", "thumbnail_url")
