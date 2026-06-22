"""Starter tests. Flesh these out — see model-server/TODO.md.

These avoid any real S3 by stubbing the model. The lifespan startup calls
model.load(), which fails gracefully (no S3 in CI) and leaves the model
unloaded — exactly the "degraded" path we assert on first.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src import model as model_module
from src.api import app


def test_health_degraded_without_model():
    # Ensure no model is loaded.
    model_module.model._estimator = None
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["model_loaded"] is False


def test_predict_503_without_model():
    model_module.model._estimator = None
    with TestClient(app) as client:
        resp = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert resp.status_code == 503


def test_predict_validation_422():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"features": [1.0, 2.0]})  # too few
    assert resp.status_code == 422


def test_predict_and_metrics_with_stub_model():
    class StubEstimator:
        def predict(self, X):
            return [0 for _ in X]

    model_module.model._estimator = StubEstimator()
    model_module.model._source = "stub://test"
    with TestClient(app) as client:
        resp = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        assert resp.status_code == 200
        assert resp.json()["predicted_class"] == 0

        metrics = client.get("/metrics").text
        assert "ml_predictions_total" in metrics
        assert "ml_predict_duration_seconds" in metrics
