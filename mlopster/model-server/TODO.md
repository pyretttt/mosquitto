# model-server — TODO

Ordered easiest → hardest. These mirror TASKS.md section 6.

## API

- [x] `GET /health` reports model load state (200 / 503).
- [x] `/metrics` exposed via instrumentator.
- [x] Custom `Counter` (predictions by class) + `Histogram` (model latency).
- [x] `POST /admin/reload-model` (Bearer token).
- [ ] Add stricter input validation on `PredictRequest` (Iris features are
      positive, < 10). Return 422 with a helpful message.
- [ ] Add a `model_source` / `model_version` field to `/health` once you load
      versioned artifacts.

## Model loading

- [x] Load artifact from S3 with a graceful "no model yet" fallback.
- [ ] Validate the loaded object actually has a `.predict` (don't trust the
      bucket). Reject incompatible artifacts.
- [ ] MLflow path: when `MLFLOW_TRACKING_URI` is set, load from the registry
      instead of S3 (`mlflow.pyfunc.load_model`). Keep S3 as the fallback.

## Tests

- [x] /health degraded, /predict 503 without a model, /predict 422 on bad input.
- [x] /predict + /metrics with a stubbed estimator.
- [ ] Add a test that exercises `storage.download_model` against a moto/MinIO
      fixture (real S3 round-trip).

## Ops

- [ ] Confirm the chart's readiness probe gates traffic until the model loads.
- [ ] Turn on the HPA (`autoscaling.enabled=true`) and load-test it.
- [ ] Structured JSON logging so logs are Loki-friendly later.
