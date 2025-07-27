"""add article_embeddings table

Revision ID: a1b2c3d4e5f6
Revises: 354820e8154c
Create Date: 2025-01-27 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '354820e8154c'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create article_embeddings table
    op.create_table(
        'article_embeddings',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('article_hash_id', sa.String(length=32), nullable=False),
        sa.Column('chunk_id', sa.String(length=64), nullable=False),
        sa.Column('embedding_vector', sa.Text(), nullable=False),
        sa.Column('text', mysql.LONGTEXT(), nullable=False),
        sa.Column('date_created', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('date_updated', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['article_hash_id'], ['articles.hash_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('article_hash_id', 'chunk_id', name='idx_article_chunk_unique')
    )
    
    # Create indexes
    op.create_index('idx_article_hash_id', 'article_embeddings', ['article_hash_id'])
    op.create_index('idx_chunk_id', 'article_embeddings', ['chunk_id'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_chunk_id', table_name='article_embeddings')
    op.drop_index('idx_article_hash_id', table_name='article_embeddings')
    
    # Drop table
    op.drop_table('article_embeddings')