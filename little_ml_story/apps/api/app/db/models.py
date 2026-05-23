from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    image_sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(16), nullable=False)
    top_class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    top_class_label: Mapped[str] = mapped_column(String(128), nullable=False)
    top_class_score: Mapped[float] = mapped_column(Float, nullable=False)
    top_k: Mapped[dict] = mapped_column(JSON, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(default=False, nullable=False)


class DailyClassCount(Base):
    """Rolled up by the Spark batch job in apps/spark_jobs/daily_aggregations.py."""

    __tablename__ = "daily_class_counts"

    day: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    class_label: Mapped[str] = mapped_column(String(128), primary_key=True)
    n: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_score: Mapped[float] = mapped_column(Float, nullable=False)
