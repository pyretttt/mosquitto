"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-23
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("image_sha256", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(16), nullable=False),
        sa.Column("top_class_id", sa.Integer, nullable=False),
        sa.Column("top_class_label", sa.String(128), nullable=False),
        sa.Column("top_class_score", sa.Float, nullable=False),
        sa.Column("top_k", sa.JSON, nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("cache_hit", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_predictions_image_sha256", "predictions", ["image_sha256"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])

    op.create_table(
        "daily_class_counts",
        sa.Column("day", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("class_label", sa.String(128), primary_key=True),
        sa.Column("n", sa.Integer, nullable=False),
        sa.Column("avg_score", sa.Float, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("daily_class_counts")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_index("ix_predictions_image_sha256", table_name="predictions")
    op.drop_table("predictions")
