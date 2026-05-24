"""init schema

Revision ID: 0001
Revises:
Create Date: 2026-05-24 00:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column("owner", sa.String(60), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_experiments_name", "experiments", ["name"])

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "experiment_id",
            sa.Integer,
            sa.ForeignKey("experiments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.String(40), nullable=False),
        sa.Column("metric_name", sa.String(40), nullable=False),
        sa.Column("metric_value", sa.Float, nullable=False),
        sa.Column("artifact_uri", sa.String(500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("experiment_id", "version", name="uq_exp_version"),
    )


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_index("ix_experiments_name", table_name="experiments")
    op.drop_table("experiments")
