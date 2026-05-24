"""Insert ~500_000 fake inference rows in batches of 50k.

Key lesson: ClickHouse ABHORS small inserts. Each insert creates a "part"
on disk; many parts -> background merges struggle -> you get the famous
"Too many parts" error. Always batch in the producer (or use async inserts).
"""

import json
import random
import time
import uuid
from datetime import datetime, timedelta, timezone

import clickhouse_connect
import numpy as np

CH_HOST = "localhost"
TOTAL_ROWS = 500_000
BATCH = 50_000

MODELS = [
    ("churn",       ["1.0.0", "1.1.0", "2.0.0"]),
    ("fraud",       ["0.9.0", "1.0.0"]),
    ("recommender", ["3.2.1", "3.3.0"]),
]
STATUSES = ["OK"] * 95 + ["ERR"] * 4 + ["TIMEOUT"] * 1  # 95% OK


def make_batch(n: int, base_ts: datetime) -> list[list]:
    rows = []
    for i in range(n):
        model_name, versions = random.choice(MODELS)
        model_version = random.choice(versions)
        # latency: log-normal — most fast, a long tail
        latency = float(np.random.lognormal(mean=3.0, sigma=0.6))
        rows.append([
            base_ts + timedelta(milliseconds=i * 17),
            model_name,
            model_version,
            uuid.uuid4().hex,
            random.randint(1, 1_000_000),
            latency,
            random.choice(STATUSES),
            float(np.random.normal(35, 10)),
            float(np.random.exponential(120.0)),
            float(np.random.beta(2, 5)),
            json.dumps({"k1": random.random(), "k2": random.random()}),
        ])
    return rows


def main() -> None:
    client = clickhouse_connect.get_client(host=CH_HOST, database="mlops", user="mlflow", password="mlflow")
    columns = [
        "ts", "model_name", "model_version", "request_id", "user_id",
        "latency_ms", "status", "feature_age", "feature_amount",
        "prediction", "raw_features",
    ]

    now = datetime.now(tz=timezone.utc)
    inserted = 0
    t0 = time.time()
    while inserted < TOTAL_ROWS:
        n = min(BATCH, TOTAL_ROWS - inserted)
        # spread batches across the last 14 days
        base = now - timedelta(days=random.uniform(0, 14))
        client.insert("inferences", make_batch(n, base), column_names=columns)
        inserted += n
        print(f"  inserted {inserted:,} / {TOTAL_ROWS:,}")

    dt = time.time() - t0
    print(f"\nDone. {TOTAL_ROWS:,} rows in {dt:.1f}s "
          f"({TOTAL_ROWS / dt:,.0f} rows/s).")


if __name__ == "__main__":
    main()
