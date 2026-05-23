CREATE DATABASE IF NOT EXISTS mlops;

CREATE TABLE IF NOT EXISTS mlops.prediction_events
(
    event_time      DateTime64(3) DEFAULT now64(3),
    request_id      UUID,
    image_sha256    FixedString(64),
    model_name      LowCardinality(String),
    model_version   LowCardinality(String),
    top_class_id    UInt16,
    top_class_label LowCardinality(String),
    top_class_score Float32,
    latency_ms      Float32
)
ENGINE = MergeTree
PARTITION BY toDate(event_time)
ORDER BY (model_name, model_version, event_time)
TTL toDate(event_time) + INTERVAL 30 DAY;

CREATE TABLE IF NOT EXISTS mlops.hourly_class_counts
(
    hour            DateTime,
    model_name      LowCardinality(String),
    model_version   LowCardinality(String),
    top_class_label LowCardinality(String),
    n               UInt64,
    sum_score       Float64
)
ENGINE = SummingMergeTree
PARTITION BY toDate(hour)
ORDER BY (model_name, model_version, top_class_label, hour);

CREATE MATERIALIZED VIEW IF NOT EXISTS mlops.hourly_class_counts_mv
TO mlops.hourly_class_counts
AS SELECT
    toStartOfHour(event_time) AS hour,
    model_name,
    model_version,
    top_class_label,
    count()                   AS n,
    sum(top_class_score)      AS sum_score
FROM mlops.prediction_events
GROUP BY hour, model_name, model_version, top_class_label;
