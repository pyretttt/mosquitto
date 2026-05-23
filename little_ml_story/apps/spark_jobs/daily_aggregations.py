"""Daily rollup from ClickHouse prediction events into Postgres.

Reads yesterday's rows from `mlops.prediction_events` via the ClickHouse
JDBC driver, aggregates per-class counts and average score, then upserts
into the `daily_class_counts` table in Postgres via JDBC.

Run inside the `spark` container:

    docker compose run --rm spark spark-submit \\
        --master local[2] /app/apps/spark_jobs/daily_aggregations.py

Or override the target day:

    TARGET_DAY=2026-05-22 docker compose run --rm spark spark-submit ...
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, lit, to_timestamp


def _target_day() -> date:
    raw = os.getenv("TARGET_DAY")
    if raw:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    return (datetime.now(tz=timezone.utc) - timedelta(days=1)).date()


def main() -> None:
    day = _target_day()
    next_day = day + timedelta(days=1)

    ch_host = os.getenv("CLICKHOUSE_HOST", "clickhouse")
    ch_port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    ch_db = os.getenv("CLICKHOUSE_DATABASE", "mlops")
    ch_user = os.getenv("CLICKHOUSE_USER", "default")
    ch_pwd = os.getenv("CLICKHOUSE_PASSWORD", "")

    pg_host = os.getenv("POSTGRES_HOST", "postgres")
    pg_port = int(os.getenv("POSTGRES_PORT", "5432"))
    pg_db = os.getenv("POSTGRES_DB", "mlops")
    pg_user = os.getenv("POSTGRES_USER", "mlops")
    pg_pwd = os.getenv("POSTGRES_PASSWORD", "mlops")

    spark = (
        SparkSession.builder.appName("daily_aggregations")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    ch_url = f"jdbc:clickhouse://{ch_host}:{ch_port}/{ch_db}"
    query = (
        "(SELECT model_name, model_version, top_class_label, top_class_score, event_time "
        f"FROM {ch_db}.prediction_events "
        f"WHERE event_time >= toDateTime('{day} 00:00:00') "
        f"AND event_time < toDateTime('{next_day} 00:00:00')) AS src"
    )

    src = (
        spark.read.format("jdbc")
        .option("url", ch_url)
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver")
        .option("user", ch_user)
        .option("password", ch_pwd)
        .option("dbtable", query)
        .load()
    )

    rolled = (
        src.groupBy("top_class_label")
        .agg(count(lit(1)).alias("n"), avg("top_class_score").alias("avg_score"))
        .withColumn("day", to_timestamp(lit(day.isoformat())))
        .selectExpr("day", "top_class_label as class_label", "n", "avg_score")
    )

    pg_url = f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}"

    rolled.createOrReplaceTempView("rolled")

    rolled.write.format("jdbc").option("url", pg_url).option("driver", "org.postgresql.Driver").option(
        "user", pg_user
    ).option("password", pg_pwd).option("dbtable", "spark_daily_staging").mode("overwrite").save()

    import psycopg2  # type: ignore

    with psycopg2.connect(
        host=pg_host,
        port=pg_port,
        dbname=pg_db,
        user=pg_user,
        password=pg_pwd,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daily_class_counts (day, class_label, n, avg_score)
                SELECT day, class_label, n, avg_score FROM spark_daily_staging
                ON CONFLICT (day, class_label) DO UPDATE
                  SET n = EXCLUDED.n,
                      avg_score = EXCLUDED.avg_score;
                DROP TABLE spark_daily_staging;
                """
            )
        conn.commit()

    print(f"[spark] daily_aggregations complete for {day.isoformat()}")
    spark.stop()


if __name__ == "__main__":
    main()
