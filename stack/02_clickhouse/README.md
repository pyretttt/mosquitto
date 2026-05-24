# 02 — ClickHouse (≈ 45 min)

You will store **millions of fake ML inference events** in ClickHouse and run analytics on them (p95 latency per model, error rate per hour, feature drift counters). This is exactly what production ML monitoring systems do (Evidently, Arize, internal "predictions" tables, etc.).

## Concepts you must be able to explain

1. **Row store vs column store.** Postgres stores all columns of a row together — great for OLTP (fetch one user, write one order). ClickHouse stores each column separately — great for OLAP (sum 1 column across 100 M rows). Aggregations skip columns they don't read entirely.
2. **MergeTree family** — the workhorse engine. `ORDER BY` defines the sort key (NOT a unique key) and is the primary index. Data is written into "parts" and later merged in the background.
3. **Why ClickHouse is fast** — vectorised execution, compression per column (LZ4/ZSTD), data skipping using sparse primary index + min/max indexes per granule (8192 rows).
4. **PARTITION BY** — usually by `toYYYYMM(ts)` or `toDate(ts)`. Lets `DROP PARTITION` instead of `DELETE` (cheap retention).
5. **Materialised views** = "insert triggers" that aggregate on write. Heavily used to pre-aggregate metrics (e.g. p95 latency per minute).
6. **Async inserts** (`async_insert=1`) — ClickHouse hates many small inserts. Either you batch in the app, or you turn on async_insert and let the server batch.
7. **No UPDATE/DELETE in the OLTP sense.** `ALTER TABLE ... DELETE` is a mutation, async, expensive. Use `ReplacingMergeTree` if you need "latest version wins" semantics.

## Task

### Step 1 — Start ClickHouse

```bash
docker compose up -d
docker compose logs -f clickhouse  # wait until "Ready for connections"
```

Open the web UI: <http://localhost:8123/play>

Try the SQL: `SELECT version()` — you should see something like `24.x`.

### Step 2 — Install Python deps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3 — Create the schema and load fake data

```bash
python scripts/01_create_schema.py
python scripts/02_seed_inferences.py   # inserts ~500k rows in batches
```

Read `02_seed_inferences.py` — notice we **batch** 50 000 rows per insert. This is the #1 rule of ClickHouse.

### Step 4 — Run the analytics queries

```bash
python scripts/03_run_analytics.py
```

It runs three queries you'll absolutely be asked about in interviews:

1. **p95 / p99 latency per model_version per hour** — `quantilesExact(0.95, 0.99)(latency_ms)`.
2. **Error rate per model_version** — share of rows where `status != 'OK'`.
3. **"Drift" proxy**: per-day mean of one feature value, to demonstrate `GROUP BY` over a wide table.

Then open the Play UI and add your own:
```sql
SELECT model_version, count() FROM inferences GROUP BY model_version ORDER BY count() DESC;
```

### Step 5 — Look at the table internals

In the Play UI:
```sql
SELECT
    table, name, rows, marks,
    formatReadableSize(bytes_on_disk) AS size,
    formatReadableSize(data_uncompressed_bytes) AS raw,
    round(data_compressed_bytes / data_uncompressed_bytes, 3) AS ratio
FROM system.parts
WHERE table = 'inferences' AND active;
```

You will see a **compression ratio of ~10–20×**. That's the column store paying off — the `model_name` column has very few distinct values so LZ4 squashes it.

### Step 6 — Materialised view (optional, +15 min)

Run `scripts/04_create_mv.py`. It creates a `SummingMergeTree` table + a materialised view that maintains rolled-up "count per minute per model" automatically as new rows arrive. Now a dashboard reading the rollup is microseconds instead of seconds.

## Interview questions to rehearse

- "Why ClickHouse for inference logs and not Postgres?" → Postgres is fine until ~10 M rows; after that, aggregations get slow because it has to read every row. CH reads only the column you aggregate, with massive compression.
- "What's the difference between `MergeTree` and `ReplacingMergeTree`?"
- "I have 1000 services each sending 1 row/sec. What's wrong with that pattern?" → 1000 inserts/sec of 1 row creates many tiny parts → merge storm → server falls over. Fix: async inserts, Kafka in front of CH, or batch in the producer.
- "How do you delete old data efficiently?" → `PARTITION BY toYYYYMM(ts)` then `ALTER TABLE ... DROP PARTITION`. Or TTL clauses.
- "Why is `ORDER BY (model_version, ts)` different from `ORDER BY (ts, model_version)`?" → The first lets queries filtered by `model_version` skip lots of data using the sparse index; the second doesn't.

## References

- ClickHouse "core concepts": <https://clickhouse.com/docs/en/intro>
- MergeTree engine: <https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/mergetree>
- "How ClickHouse uses indexes": <https://clickhouse.com/docs/en/optimize/sparse-primary-indexes>
- Async inserts: <https://clickhouse.com/docs/en/optimize/asynchronous-inserts>
- Materialised views explained: <https://clickhouse.com/docs/en/materialized-view>
- `clickhouse-connect` Python client: <https://clickhouse.com/docs/integrations/python>
