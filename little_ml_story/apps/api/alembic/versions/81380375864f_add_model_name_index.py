"""add model_name index

Revision ID: 81380375864f
Revises: 0001_initial
Create Date: 2026-05-23 13:30:30.971293
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '81380375864f'
down_revision: Union[str, Sequence[str], None] = '0001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_predictions_model_name_created_at",
        "predictions",
        ["model_name", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_model_name_created_at", table_name="predictions")