"""Materialised view = automatic incremental rollup on insert.

Pattern: create a SummingMergeTree target table, then a materialised view that
SELECTs from `inferences` and INSERTs into the rollup. Every new INSERT to
`inferences` triggers the SELECT and appends to the rollup. Queries on the
rollup are tiny -> dashboards become microseconds instead of seconds.
"""

import clickhouse_connect

CH_HOST = "localhost"

ROLLUP_TABLE = """CREATE TABLE IF NOT EXISTS inferences_per_minute
(
    minute        DateTime,
    model_version LowCardinality(String),
    n             UInt64,
    errors        UInt64,
    sum_latency   Float64
)
ENGINE = SummingMergeTree
ORDER BY (model_version, minute);"""

MV = """CREATE MATERIALIZED VIEW IF NOT EXISTS inferences_per_minute_mv
TO inferences_per_minute AS
SELECT
    toStartOfMinute(ts) AS minute,
    model_version,
    count() AS n,
    countIf(status != 'OK') AS errors,
    sum(latency_ms) AS sum_latency
FROM mlops.inferences
GROUP BY minute, model_version;"""

CHECK = """SELECT minute, model_version, n, errors,
       round(sum_latency / n, 2) AS mean_latency
FROM mlops.inferences_per_minute
ORDER BY minute DESC
LIMIT 5;"""


def main() -> None:
    client = clickhouse_connect.get_client(host=CH_HOST, database="mlops", user="mlflow", password="mlflow")
    client.command(ROLLUP_TABLE)
    client.command(MV)
    print("MV created. Note: it only sees data inserted AFTER creation.")
    print("Re-run scripts/02_seed_inferences.py to see new rows flow in.\n")

    res = client.query(CHECK)
    print(" | ".join(res.column_names))
    for row in res.result_rows:
        print(" | ".join(str(c) for c in row))


if __name__ == "__main__":
    main()
