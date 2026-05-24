"""Pretend to be a FastAPI service. Emit ~500 structured log events.

Each "request" produces:
   request_started -> inference_done   (95% of the time)
                   -> error             ( 5% of the time)

All events of a request share the same `request_id` so you can reconstruct
the trace by grouping on it.

In production you would NOT call OpenSearch from your hot path. Instead the
app prints JSON to stdout, and Filebeat / Fluent Bit / Vector tails the
container logs and ships to OpenSearch. Here we use the Python client
directly for brevity.
"""

import random
import time
import uuid
from datetime import datetime, timezone

from opensearchpy import OpenSearch, helpers

INDEX = f"logs-mlops-{datetime.now(timezone.utc):%Y.%m.%d}"
MODELS = [("churn", "1.0.0"), ("churn", "2.0.0"), ("fraud", "0.9.0")]
N_REQUESTS = 500

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
)


def _doc(level: str, event: str, **fields) -> dict:
    return {
        "_index": INDEX,
        "_source": {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "service": "predictor",
            "event": event,
            **fields,
        },
    }


def fake_request() -> list[dict]:
    request_id = uuid.uuid4().hex
    user_id = random.randint(1, 10_000)
    model_name, model_version = random.choice(MODELS)
    latency = max(2.0, random.lognormvariate(3.0, 0.7))
    is_error = random.random() < 0.05

    docs = [
        _doc(
            "INFO", "request_started",
            request_id=request_id, user_id=user_id,
            model_name=model_name, model_version=model_version,
            message=f"request started for user {user_id}",
        ),
    ]
    if is_error:
        docs.append(_doc(
            "ERROR", "error",
            request_id=request_id, user_id=user_id,
            model_name=model_name, model_version=model_version,
            latency_ms=latency, status_code=500,
            message="model inference failed",
            exception=random.choice([
                "TimeoutError: triton call took 5000ms",
                "ValueError: feature 'amount' is NaN",
                "ConnectionResetError: feature store reset",
            ]),
        ))
    else:
        docs.append(_doc(
            "INFO", "inference_done",
            request_id=request_id, user_id=user_id,
            model_name=model_name, model_version=model_version,
            latency_ms=latency, status_code=200,
            message="inference done",
        ))
    return docs


def main() -> None:
    print(f"streaming into index {INDEX} ...")
    actions = []
    for _ in range(N_REQUESTS):
        actions.extend(fake_request())

    # `helpers.bulk` batches into a single _bulk request -> way faster.
    ok, errors = helpers.bulk(client, actions, stats_only=False)
    print(f"indexed {ok} docs, errors={errors}")
    client.indices.refresh(index=INDEX)
    count = client.count(index=INDEX)["count"]
    print(f"index now has {count} docs")
    time.sleep(0.2)


if __name__ == "__main__":
    main()
