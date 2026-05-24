# 05 — PySpark + MLlib (≈ 60 min)

You will run PySpark locally (one container) and build an end-to-end pipeline: load CSV → feature engineering → train a Logistic Regression → evaluate → save the model. You will then **look at the query plan** with `df.explain()` and understand how Spark turns your Python into a DAG.

## Concepts you must be able to explain

1. **Driver vs executors.** The driver is your `pyspark` process — it builds the DAG, ships tasks. Executors are JVM processes that actually crunch data on partitions. In `local[4]` mode you have 4 executor threads on one machine.
2. **RDD vs DataFrame vs Dataset.** RDD = low-level, untyped, no Catalyst optimiser. DataFrame = table abstraction with Catalyst (the SQL optimiser) + Tungsten (off-heap binary memory). Always prefer DataFrame/SQL in PySpark.
3. **Lazy evaluation.** `df.filter(...).select(...)` builds a plan; **nothing runs** until an *action* (`.count()`, `.collect()`, `.write...`). Catalyst can re-order, push predicates, prune columns.
4. **Transformations: narrow vs wide.** `select`, `filter`, `map` are narrow (no shuffle). `groupBy`, `join`, `distinct`, `orderBy` are wide (shuffle = data moves across the network between executors).
5. **Partitioning.** Number of partitions ≈ parallelism. Too few → executors idle. Too many → tiny tasks, overhead dominates. Default after a shuffle = `spark.sql.shuffle.partitions` (200 — usually too high locally).
6. **MLlib pipelines.** `Pipeline([stage1, stage2, ...])` makes featurisation + model a single object you can `.fit()` and `.transform()`. Same object goes to scoring → train/serve skew gone.
7. **Where MLlib fits.** When your data doesn't fit on one machine. For small data (single-node), use sklearn; for huge data, MLlib + horizontal cluster. Cost: MLlib has fewer algorithms than sklearn and is slower per-row.

## Task

### Step 1 — Start the Spark container

```bash
docker compose up -d
docker compose logs -f spark   # waits ~10s
```

The container has Spark 3.5 + Python + the sample CSV already mounted.

### Step 2 — Open a shell inside the container

```bash
docker compose exec spark bash
```

You'll be in `/work`. Run:

```bash
python scripts/01_explore.py
```

This loads `data/titanic.csv`, prints schema, shows count + a few `groupBy` examples, and calls `.explain(True)` — read the output. You'll see the **logical plan → optimised plan → physical plan**. Notice how `select` was pushed before `filter`.

### Step 3 — Run the ML pipeline

```bash
python scripts/02_pipeline.py
```

It builds a Pipeline with: `StringIndexer` (Sex) → `OneHotEncoder` (Pclass) → `Imputer` (Age) → `VectorAssembler` → `LogisticRegression`. Splits 80/20, fits, prints AUC on the holdout, saves the fitted Pipeline to `models/titanic_lr/`.

### Step 4 — Load the saved model and score new rows

```bash
python scripts/03_score.py
```

This loads the saved Pipeline and scores 3 hand-crafted rows — proves the model is a self-contained artifact you can ship.

### Step 5 — Look at the Spark UI

While `02_pipeline.py` is running, open <http://localhost:4040> in your browser. You'll see jobs → stages → tasks. Find a "shuffle write" — that's where the wide transformation happened.

### Step 6 — (Optional, +10 min) Repartition exercise

In `01_explore.py` add at the end:

```python
print("default partitions:", df.rdd.getNumPartitions())
df2 = df.repartition(8)
print("after repartition:", df2.rdd.getNumPartitions())
df3 = df.coalesce(1)
print("after coalesce:   ", df3.rdd.getNumPartitions())
```

`repartition` reshuffles, `coalesce` only merges — that's the kind of subtlety asked at interviews.

## Interview questions to rehearse

- "What's the difference between `repartition` and `coalesce`?" → `repartition(n)` is a full shuffle; `coalesce(n)` only merges existing partitions, no shuffle, but cannot increase partitions efficiently.
- "What's a shuffle and why is it expensive?" → moving data across the network so rows with the same key end up on the same executor. Disk + network IO.
- "Why did my Spark job OOM at the executor?" → typically skew (one key has 90% of rows) or `collect()` on a huge DF. Mitigations: salt the key, increase shuffle partitions, use `broadcast` join for small side.
- "When would you use Spark vs Pandas vs Polars?" → Pandas/Polars: single machine, < ~50 GB. Spark: doesn't fit on one machine, or you need a unified batch + streaming engine.
- "What does Catalyst do?" → rule-based + cost-based optimiser. Examples: predicate pushdown (filters move before joins), column pruning (only requested columns are read from Parquet), constant folding.
- "Spark MLlib vs scikit-learn pickled in Spark UDFs?" → MLlib trains distributedly; sklearn in UDF can do *inference* distributedly but training is single-node. Many teams train in sklearn and serve in Spark with a `mlflow.pyfunc` UDF.

## References

- Spark "quick start": <https://spark.apache.org/docs/latest/quick-start.html>
- DataFrame API reference: <https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql.html>
- MLlib pipelines guide: <https://spark.apache.org/docs/latest/ml-pipeline.html>
- "Mastering Spark SQL" (free book): <https://books.japila.pl/spark-sql-internals/>
- Skew + shuffle tuning: <https://spark.apache.org/docs/latest/sql-performance-tuning.html>
