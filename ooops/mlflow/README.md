# MLflow tracking server

- Backend store: SQLite at `/mlflow/db/mlflow.db` (volume: `mlflow-db`).
- Artifact store: local FS at `/mlflow/artifacts` (volume: `mlflow-artifacts`).
- `--serve-artifacts` means clients download artifacts through the server, so
  the `ml-app` container does **not** need to share the artifact volume. This
  also makes the k8s migration cleaner — only one pod touches the artifact PV.

Open the UI: <http://localhost:5001>

## TODO(you)

1. Swap the backend store to PostgreSQL (add a `mlflow-db` Postgres service).
   SQLite is fine for a demo, not for concurrent writes.
2. Swap the artifact store to MinIO (S3-compatible) so the k8s migration is
   trivial. With MinIO you'll set `--artifacts-destination s3://bucket/`.
3. Turn on basic auth (`MLFLOW_AUTH_CONFIG_PATH`) — even on localhost it's
   worth the 5 minutes to learn how.
