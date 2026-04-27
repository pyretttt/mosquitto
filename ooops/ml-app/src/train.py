"""Train a tiny sklearn classifier on the Iris dataset and log to MLflow.

Runs once, as a script:

    python -m src.train

Intentionally minimal — this is the "hello world" of model training. Real work
for you to add (see docs/LEARNING_PATH.md):

  - Hyperparameter sweep (log multiple runs, compare).
  - Proper train/val/test split + metrics beyond accuracy.
  - Data versioning (DVC, lakeFS, etc.).
  - Move this script out of the API container into a CI job.
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .config import settings

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def train() -> str:
    """Train, log to MLflow, register the model. Returns the run_id."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    params = {"n_estimators": 100, "max_depth": 4, "random_state": 42}

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
        log.info("trained model: accuracy=%.4f f1=%.4f", acc, f1)

        # `registered_model_name` creates/updates an entry in the Model Registry.
        # First run -> version 1, next run -> version 2, etc.
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=settings.model_name,
            input_example=X_train.head(1),
        )
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{settings.model_name}'")

        latest = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name=settings.model_name, version=latest.version, stage=settings.model_stage
        )
        return run.info.run_id


if __name__ == "__main__":
    run_id = train()
    log.info("logged run %s to %s", run_id, settings.mlflow_tracking_uri)
