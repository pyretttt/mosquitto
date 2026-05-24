"""SQLAlchemy ORM models.

This file declares the schema in Python; Alembic uses these classes to
generate migrations. We never call Base.metadata.create_all() in production
because it cannot evolve the schema safely — Alembic can.
"""

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    owner: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        # The MOST important constraint of a model registry:
        # you cannot register the same (experiment, version) twice.
        UniqueConstraint("experiment_id", "version", name="uq_exp_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"),
        index=True
    )
    version: Mapped[str] = mapped_column(String(40))
    metric_name: Mapped[str] = mapped_column(String(40))
    metric_value: Mapped[float] = mapped_column(Float)
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    experiment: Mapped[Experiment] = relationship(back_populates="versions")
