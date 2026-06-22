"""Training DAG: pull data from S3 -> train -> push artifact to S3.

This is the seam Airflow owns. The model server only *serves* artifacts; it
never trains. When MLflow lands, this DAG will additionally log params/metrics
and register the model (see the TODO markers).

Delivered to Airflow via git-sync (airflow.dags.gitSync in the chart values).
S3 credentials come from the `s3-credentials` Secret, surfaced as env vars on
the worker pods (airflow.extraEnvFrom in the chart values).

TODO(you):
  - point `_load_dataset` at a real dataset in the `data` bucket instead of the
    bundled sklearn Iris.
  - add an MLflow logging task once the tracking server is enabled.
  - tune the schedule; it's manual-only for now.
"""

from __future__ import annotations

import datetime
import logging
import os
import tempfile

from airflow.decorators import dag, task

log = logging.getLogger(__name__)

MODELS_BUCKET = os.environ.get("S3_MODELS_BUCKET", "models")
DATA_BUCKET = os.environ.get("S3_DATA_BUCKET", "data")
MODEL_KEY = os.environ.get("MODEL_KEY", "model.joblib")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "http://storage-minio.storage:9000")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _s3_client():
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=AWS_REGION,
        config=Config(signature_version="s3v4"),
    )


@dag(
    dag_id="train_iris_model",
    schedule=None,  # manual trigger for now
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    tags=["mlopster", "training"],
    default_args={"retries": 1, "retry_delay": datetime.timedelta(minutes=2)},
)
def train_iris_model():
    @task
    def load_dataset() -> str:
        """Load data and stash it as a parquet/csv in /tmp for the next task.

        TODO(you): replace the sklearn bundled dataset with a real pull from the
        `data` bucket (s3.download_file(DATA_BUCKET, "iris.csv", ...)).
        """
        from sklearn.datasets import load_iris

        data = load_iris(as_frame=True).frame
        path = tempfile.mktemp(suffix=".csv")
        data.to_csv(path, index=False)
        log.info("dataset written to %s (%d rows)", path, len(data))
        return path

    @task
    def train(dataset_path: str) -> str:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        df = pd.read_csv(dataset_path)
        target_col = "target"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        log.info("trained model, train accuracy=%.3f", clf.score(X, y))

        # TODO(you, MLflow): mlflow.log_params/metrics + mlflow.sklearn.log_model
        # and register the model instead of (or in addition to) writing to S3.

        import joblib

        out = tempfile.mktemp(suffix=".joblib")
        joblib.dump(clf, out)
        return out

    @task
    def upload(model_path: str) -> None:
        s3 = _s3_client()
        s3.upload_file(model_path, MODELS_BUCKET, MODEL_KEY)
        log.info("uploaded model -> s3://%s/%s", MODELS_BUCKET, MODEL_KEY)
        # The running model server won't pick this up until it reloads. Either
        # hit POST /admin/reload-model or let a rollout restart do it.

    upload(train(load_dataset()))


train_iris_model()
