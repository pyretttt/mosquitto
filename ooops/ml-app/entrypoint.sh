#!/usr/bin/env sh
set -e

# =============================================================================
# PROBLEM (read this before "fixing" the script)
# =============================================================================
# When `docker compose up` starts everything at once, ml-app and mlflow boot
# in parallel. Compose's `depends_on: [mlflow]` only waits for the mlflow
# *container to be created* — it does NOT wait for the MLflow HTTP server
# inside that container to actually accept requests.
#
# Result: on a cold start, ml-app reaches this script ~1–2 seconds before
# MLflow is listening on :5000. Then either of the steps below explodes with
# something like:
#
#     mlflow.exceptions.MlflowException: API request to ... failed
#     urllib3.exceptions.MaxRetryError: ... Connection refused
#
# That's the failure mode you'll see if you don't fix this.
#
# =============================================================================
# TODO(you): make this script wait for MLflow to be reachable
# =============================================================================
# Pick ONE of these approaches and implement it. Each teaches you something
# different — read up on the trade-offs before you choose.
#
#   Option A — poll from this script.
#     Loop with curl/wget/python until GET http://mlflow:5000/health returns
#     200, with a timeout. Pros: self-contained. Cons: every container
#     re-implements the wait.
#
#   Option B — Docker healthcheck + condition: service_healthy.
#     Add a HEALTHCHECK to the mlflow service in docker-compose.yml, then
#     change ml-app's depends_on to:
#         depends_on:
#           mlflow:
#             condition: service_healthy
#     Pros: declarative, works for every dependent service. Cons: requires
#     curl/wget in the mlflow image (the python:3.11-slim base has neither —
#     either install one in mlflow/Dockerfile, or healthcheck via python).
#
#   Option C — wait-for-it.sh / dockerize.
#     Drop in a generic wait script. Pros: zero new code. Cons: another
#     dependency to vendor.
#
# Hint: MLflow exposes GET /health → 200 OK once it's ready.
# =============================================================================

python - <<'PY'
import os, time, urllib.request, urllib.error
uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
for i in range(10):
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

# Train on first boot so /predict works out of the box. Once you move
# training into CI (LEARNING_PATH step 9), delete the line below.
# Until you implement the wait above, this WILL fail on a clean start —
# that's the symptom that should drive you to fix it. ✅
# python -m src.train || echo "training failed (likely mlflow not ready) — API will start in degraded mode"

exec uvicorn src.api:app --host 0.0.0.0 --port 8000
