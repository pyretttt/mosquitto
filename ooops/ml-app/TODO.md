# ml-app — TODO

Roughly ordered easiest → hardest. Tick them off as you go.

## API surface

- [ ] `GET /health` route in `src/api.py`. Returns 200 + JSON `{status, model_loaded, model_source}` when ready, 503 when the model failed to load.
- [ ] Wire `prometheus-fastapi-instrumentator` so `/metrics` is exposed. (Already in `requirements.txt`.) Verify with `curl localhost:8000/metrics` after `docker compose up`.
- [ ] Add a custom `Counter` for predictions, labelled by `predicted_class`. Increment it inside `/predict`. Verify it shows up in `/metrics`.
- [ ] Add a custom `Histogram` (`ml_predict_duration_seconds`) timing only the `model.predict(...)` call (not HTTP overhead). Use `.time()` as a context manager.
- [ ] Add an authenticated `POST /admin/reload` route that calls `model.load()` so the API can pick up new model versions without restarting.

## Validation & errors

- [ ] Add input range validation in `PredictRequest` (Iris features are all positive, < 10). Return 422 on bad input.
- [ ] Add a `RequestValidationError` exception handler that emits a structured log line.

## Tests

- [ ] Test `/health` (200 ready, 503 not ready).
- [ ] Test `/metrics` returns Prometheus exposition format and contains your custom counter name *after* a `/predict` call.
- [ ] Test `/predict` rejects wrong feature count → 422, non-numeric → 422.
- [ ] Add a real training test: spin up MLflow with `MLFLOW_TRACKING_URI=file:./mlruns`, run `train()`, assert a model version was registered.

## Boot/runtime

- [ ] Fix the MLflow start-up race in `entrypoint.sh` (see the comment block in that file).
- [ ] Add a Docker `HEALTHCHECK` to the `ml-app` service in `docker-compose.yml` that calls your new `/health`. Then change `depends_on` on consumers (e.g. CI smoke tests) to `condition: service_healthy`.

## Model lifecycle

- [ ] Use the MLflow Model Registry properly: promote a version to `Staging` in the UI, set `ML_MODEL_STAGE=Staging`, restart the API, confirm `model_source` reflects the new URI.
- [ ] Move `python -m src.train` out of `entrypoint.sh` and into the GitLab CI `train` stage (LEARNING_PATH step 9). The inference container should not know how to train.

## Stretch

- [ ] Replace `RandomForestClassifier` with a small neural net (or anything else) and confirm nothing else needs to change — that's the value of going through `mlflow.pyfunc.load_model`.
- [ ] Add structured JSON logging (`python-json-logger`) so logs are usable by Loki later.
- [ ] Add OpenTelemetry tracing and ship traces to Tempo/Jaeger (bonus: instrument the model call as its own span).
