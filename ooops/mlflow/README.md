# MLflow tracking server

- Backend store: PostgreSQL in the `mlflow-db` service.
- Artifact store: MinIO/S3 in the `mlflow-artifacts` service, bucket `mlflow`.
- `--serve-artifacts` means clients download artifacts through the server, so
  the `ml-app` container does **not** need direct artifact-store access.

Open the UI: <http://localhost:5001>
Open the MinIO console: <http://localhost:9001>

## TODO(you)

1. Turn on basic auth (`MLFLOW_AUTH_CONFIG_PATH`) — even on localhost it's
   worth the 5 minutes to learn how.
