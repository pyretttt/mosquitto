"""Small config module. Everything comes from env vars so compose/K8s can drive it."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classifier")
    model_name: str = os.getenv("ML_MODEL_NAME", "iris-classifier")
    # "None" is the MLflow default stage for a freshly logged model.
    # Promote to "Staging" / "Production" via the MLflow UI (exercise!).
    model_stage: str = os.getenv("ML_MODEL_STAGE", "None")


settings = Settings()
