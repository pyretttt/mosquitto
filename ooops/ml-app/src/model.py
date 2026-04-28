"""Loads the registered model from MLflow. Kept behind a small class so the API
can be written without knowing about MLflow."""

from __future__ import annotations

import logging
import threading
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from .config import settings

log = logging.getLogger(__name__)

# Must match sklearn `load_iris(..., as_frame=True).data.columns` — MLflow
# records this schema when the model is logged with that input_example.
IRIS_FEATURE_COLUMNS: tuple[str, ...] = (
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
)


class Model:
    """Thread-safe lazy loader for the registered model.

    A real deployment would have explicit reload triggers (e.g. a webhook from
    MLflow when a new version is promoted). Here we just load once at startup
    and expose a `reload()` for you to call from a route.
    TODO(you): implement an authenticated /admin/reload route.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._uri: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def source_uri(self) -> str | None:
        return self._uri

    def load(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        # "models:/<name>/<stage-or-version>" — stage "None" means "latest
        # registered version regardless of stage", which is what we want on
        # first boot. Promote to Staging/Production in the UI as an exercise.
        uri = f"models:/{settings.model_name}/latest"
        with self._lock:
            log.info("loading model from %s", uri)
            self._model = mlflow.pyfunc.load_model(uri)
            self._uri = uri
            log.info("model loaded")

    def predict(self, rows: list[list[float]]) -> list[int]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        frame = pd.DataFrame(rows, columns=list(IRIS_FEATURE_COLUMNS))
        preds = self._model.predict(frame)
        return [int(p) for p in preds]


model = Model()
