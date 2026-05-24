"""Three queries you should be able to write blindfolded."""

import clickhouse_connect

CH_HOST = "localhost"

Q_LATENCY = """SELECT
    model_version,
    toStartOfHour(ts) AS hour,
    count() AS n,
    round(quantileExact(0.50)(latency_ms), 2) AS p50,
    round(quantileExact(0.95)(latency_ms), 2) AS p95,
    round(quantileExact(0.99)(latency_ms), 2) AS p99
FROM mlops.inferences
WHERE ts >= now() - INTERVAL 7 DAY
GROUP BY model_version, hour
ORDER BY hour DESC, model_version
LIMIT 10;"""

Q_ERROR_RATE = """SELECT
    model_version,
    count() AS n,
    countIf(status != 'OK') AS errors,
    round(errors / n, 4) AS error_rate
FROM mlops.inferences
WHERE ts >= now() - INTERVAL 7 DAY
GROUP BY model_version
ORDER BY error_rate DESC;"""

Q_DRIFT = """SELECT
    toDate(ts) AS day,
    model_version,
    round(avg(feature_amount), 2) AS mean_amount,
    round(stddevSamp(feature_amount), 2) AS std_amount
FROM mlops.inferences
WHERE ts >= now() - INTERVAL 7 DAY
GROUP BY day, model_version
ORDER BY day DESC, model_version
LIMIT 10;"""


def run(client, title: str, sql: str) -> None:
    print(f"\n=== {title} ===")
    result = client.query(sql)
    print("  " + " | ".join(result.column_names))
    for row in result.result_rows:
        print("  " + " | ".join(str(c) for c in row))


def main() -> None:
    client = clickhouse_connect.get_client(host=CH_HOST, database="mlops", user="mlflow", password="mlflow")
    run(client, "p50/p95/p99 latency per model_version per hour", Q_LATENCY)
    run(client, "error rate per model_version (last 7 days)", Q_ERROR_RATE)
    run(client, "feature_amount drift proxy per day", Q_DRIFT)


if __name__ == "__main__":
    main()
