# Airflow (training)

DAGs live in `dags/`. They are delivered to the cluster via **git-sync**
(configured in `charts/orchestration/values.yaml` → `airflow.dags.gitSync`),
so committing a DAG here makes it appear in Airflow after the next sync.

## DAGs

- `train_model.py` — `train_iris_model`: load data → train RandomForest →
  upload the artifact to the `models` bucket in S3. Manual trigger for now.

## Getting the Python deps onto the workers

Two options (see `requirements.txt`):

1. **Custom image (recommended):** build `FROM apache/airflow:<version>` with
   the deps baked in, push to your registry, and set the image in chart values.
   Reproducible and fast.
2. **Runtime install:** set `airflow.extraPipPackages`. Simple, but slows every
   task pod and is not reproducible. Fine for first experiments only.

## Credentials

The DAG reads S3 via env vars from the `s3-credentials` Secret
(`airflow.extraEnvFrom` in chart values). Copy that Secret into the
orchestration namespace first (see `charts/storage` NOTES / `TASKS.md` §8).

## Running it

```bash
mise run pf-airflow      # http://localhost:8080  (admin/admin)
# Unpause "train_iris_model" and trigger it.
```

After a successful run, the `models/model.joblib` object exists. Tell the model
server to pick it up:

```bash
curl -XPOST -H "Authorization: Bearer <ADMIN_TOKEN>" \
  http://localhost:8000/admin/reload-model
```

## Later: MLflow

When MLflow is enabled (TASKS.md §5), the `train` task will also log
params/metrics and register the model, and the model server will load from the
registry instead of S3. The seams (`MLFLOW_TRACKING_URI`, TODO markers) are
already in place.
