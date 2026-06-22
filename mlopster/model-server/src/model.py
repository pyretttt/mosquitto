"""Model loading + inference.

Strategy:
  1. On load(), pull the artifact from S3 (MinIO) and deserialize with joblib.
  2. If there is no artifact yet (Airflow hasn't trained one), stay "not loaded"
     so /health reports degraded and readiness keeps traffic away — rather than
     crashing the pod.

TODO(you): when MLflow lands, add a branch here that loads from the MLflow
registry (mlflow.pyfunc.load_model) when settings.mlflow_tracking_uri is set.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading

import joblib

from . import storage

log = logging.getLogger(__name__)


class Model:
    def __init__(self) -> None:
        self._estimator = None
        self._source: str | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._estimator is not None

    @property
    def source_uri(self) -> str | None:
        return self._source

    def load(self) -> bool:
        """(Re)load the model from S3. Thread-safe so /admin/reload-model is safe."""
        with self._lock:
            fd, tmp_path = tempfile.mkstemp(suffix=".joblib", dir="/tmp")
            os.close(fd)
            try:
                if storage.download_model(tmp_path):
                    self._estimator = joblib.load(tmp_path)
                    self._source = f"s3://{storage.settings.s3_models_bucket}/{storage.settings.model_key}"
                    log.info("model loaded from %s", self._source)
                    return True
                log.warning("no model artifact available; serving will report degraded")
                return False
            except Exception:  # noqa: BLE001 - never let a bad artifact kill the pod
                log.exception("failed to deserialize model artifact")
                return False
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def predict(self, features: list[list[float]]) -> list[int]:
        if self._estimator is None:
            raise RuntimeError("model not loaded")
        preds = self._estimator.predict(features)
        return [int(p) for p in preds]


model = Model()
