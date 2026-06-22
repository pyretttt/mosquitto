"""Runtime configuration, read from environment variables.

In-cluster these come from the model-server ConfigMap (MODEL_KEY, LOG_LEVEL,
MLFLOW_TRACKING_URI) and the s3-credentials Secret (AWS_*, S3_*). Locally they
default to values that work against `mise run pf-minio`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # S3 / MinIO
    s3_endpoint_url: str
    s3_models_bucket: str
    aws_region: str
    model_key: str

    # Misc
    log_level: str
    admin_token: str | None
    # RESERVED: unused until MLflow is enabled (see TASKS.md section 5).
    mlflow_tracking_uri: str | None


def load_settings() -> Settings:
    return Settings(
        s3_endpoint_url=os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000"),
        s3_models_bucket=os.environ.get("S3_MODELS_BUCKET", "models"),
        aws_region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        model_key=os.environ.get("MODEL_KEY", "models/model.joblib"),
        log_level=os.environ.get("LOG_LEVEL", "info"),
        admin_token=os.environ.get("ADMIN_TOKEN") or None,
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI") or None,
    )


settings = load_settings()
