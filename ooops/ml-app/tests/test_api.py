"""Smoke test for the API. Runs without MLflow: we stub out the model.

TODO(you): add tests for:
  - /predict input validation (wrong feature count, non-numeric, etc.)
  - /metrics exposes the ml_predictions_total counter after a /predict call.
  - a test that spins up an MLflow file-store and asserts train() registers a model.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src import api as api_module


class _StubModel:
    loaded = True
    source_uri = "stub://test-model"

    def predict(self, rows):
        return [0 for _ in rows]


def test_health_reports_model_status(monkeypatch):
    monkeypatch.setattr(api_module, "model", _StubModel())
    with TestClient(api_module.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_class(monkeypatch):
    monkeypatch.setattr(api_module, "model", _StubModel())
    with TestClient(api_module.app) as client:
        r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    assert r.json()["predicted_class"] == 0
