# MLOps Interview Prep Stack

A set of 7 self-contained mini-projects covering the data/ML infra technologies most often asked about in Middle Python / MLOps interviews. Each task is designed for **30–60 minutes**.

## Suggested order

The order goes from "core backend storage" → "streaming / caching" → "ML-specific systems". You can do them independently, but in this order each one builds intuition for the next:

| # | Folder | Tech | Why an MLOps engineer should know it |
|---|---|---|---|
| 1 | `01_postgres_fastapi` | PostgreSQL + FastAPI | Metadata store for model registry, experiment tracking (MLflow uses Postgres). |
| 2 | `02_clickhouse` | ClickHouse | Column store for high-volume inference logs, feature stats, monitoring metrics. |
| 3 | `03_kafka_fastapi` | Kafka + FastAPI | Event bus for feature pipelines, retraining triggers, real-time inference logging. |
| 4 | `04_redis_cache` | Redis | Online feature store, prediction caching, rate limiting on inference endpoints. |
| 5 | `05_spark_ml` | PySpark + MLlib | Distributed feature engineering and batch training over large datasets. |
| 6 | `06_opensearch_logging` | OpenSearch (ELK) | Centralised structured logs for model services; root-cause analysis. |
| 7 | `07_triton_inference` | NVIDIA Triton | The "default" production inference server: multi-framework, dynamic batching, model versioning. |

## Prerequisites

- **Docker Desktop** (or OrbStack) — every task uses `docker compose`.
- **Python 3.11+** with `venv`.
- ~10 GB free disk for images.
- Apple Silicon notes:
  - Spark and OpenSearch run natively.
  - Triton is x86 only; we use `platform: linux/amd64` (slow but works for learning).

## How to do a task

```bash
cd 01_postgres_fastapi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
# then follow the README in that folder
```

Each subproject has its own `README.md` with:

1. **Concepts** — the 5–10 things you must be able to explain.
2. **Task** — concrete steps you do with your hands.
3. **Interview questions** — what you'll likely be asked.
4. **References** — official docs + 1–2 great articles.

## What you'll be able to answer after finishing

- "How would you store model metadata and serve it via an API?" → task 1
- "Where do you store millions of inference logs per day?" → task 2
- "How does your feature pipeline get data in real time?" → task 3
- "How do you avoid recomputing features on every request?" → task 4
- "How would you train on data that doesn't fit in memory?" → task 5
- "How do you debug a model that returns 500s in prod?" → task 6
- "How would you actually serve a PyTorch / ONNX model to many clients?" → task 7

Good luck. Do the tasks with your hands — reading is not enough for these systems.
