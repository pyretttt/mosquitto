"""Create the `inferences` table.

Schema design notes (interview gold):

- ENGINE MergeTree            : the default high-performance OLAP engine.
- PARTITION BY toYYYYMM(ts)   : monthly parts -> cheap retention by DROP PARTITION.
- ORDER BY (model_version, ts): the SORT key. ClickHouse keeps a SPARSE index
  on this — one entry per 8192 rows ("granule"). Queries filtered by
  `model_version` or `model_version + ts range` skip almost all data.
- LowCardinality(String)      : dict-encoded string. Use it for any column
  with < ~10k distinct values (model name, status, region).
- CODEC(ZSTD(3)) on JSON      : ZSTD compresses JSON much better than LZ4.
"""

import clickhouse_connect

CH_HOST = "localhost"

DDL = """
CREATE TABLE IF NOT EXISTS inferences
(
    ts              DateTime64(3, 'UTC'),
    model_name      LowCardinality(String),
    model_version   LowCardinality(String),
    request_id      String,
    user_id         UInt64,
    latency_ms      Float32,
    status          LowCardinality(String),     -- 'OK' | 'ERR' | 'TIMEOUT'
    feature_age     Float32,                    -- example numeric feature
    feature_amount  Float32,                    -- example numeric feature
    prediction      Float32,
    raw_features    String CODEC(ZSTD(3))       -- JSON blob, heavy column
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (model_version, ts)
SETTINGS index_granularity = 8192;
"""


def main() -> None:
    client = clickhouse_connect.get_client(host=CH_HOST, database="mlops", user="mlflow", password="mlflow")
    client.command(DDL)
    print("OK — table `inferences` is ready.")


if __name__ == "__main__":
    main()
