#!/usr/bin/env sh
set -e

# Wait for MLflow to accept connections. Crude but effective.
python - <<'PY'
import os, time, urllib.request, urllib.error
uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
for i in range(60):
    try:
        urllib.request.urlopen(f"{uri}/health", timeout=1)
        print(f"mlflow reachable at {uri}")
        break
    except Exception as e:
        print(f"waiting for mlflow ({i}): {e}")
        time.sleep(1)
else:
    raise SystemExit("mlflow did not come up in time")
PY

# Train on startup so /predict works out of the box. Idempotent: each run is a
# new MLflow run; the latest version wins in load_model().
# TODO(you): move training out of the API container into the CI pipeline.
python -m src.train || echo "training failed, API will start in degraded mode"

exec uvicorn src.api:app --host 0.0.0.0 --port 8000
