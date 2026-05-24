# 01 — PostgreSQL + FastAPI (≈ 45 min)

You build a tiny **model registry**: a service that stores metadata about trained ML models (name, version, metrics, artifact path) in PostgreSQL and exposes it via FastAPI.

This is the same role Postgres plays inside MLflow, Weights & Biases self-hosted, Kubeflow Metadata, Feast registry, etc.

## Concepts you must be able to explain

1. **ACID** — what each letter means and *why a model registry needs them* (e.g. you must not register version 5 twice).
2. **Transaction isolation levels** — `READ COMMITTED` (default in Postgres) vs `REPEATABLE READ` vs `SERIALIZABLE`. What problem does each prevent (dirty read, non-repeatable read, phantom read)?
3. **Indexes** — B-tree vs GIN/GiST. When does `EXPLAIN ANALYZE` show a `Seq Scan` and why is that often bad?
4. **Connection pooling** — why opening a fresh connection per HTTP request is a disaster. `asyncpg` pool vs PgBouncer.
5. **Sync vs async drivers** — `psycopg2` (sync) vs `asyncpg` / `psycopg[binary,pool] async` vs SQLAlchemy 2.0 async.
6. **Migrations** — why you never run `Base.metadata.create_all()` in production. Alembic.
7. **N+1 problem** — what it is in ORM, how `selectinload` / `joinedload` fixes it.

## Task

### Step 1 — Start Postgres

```bash
docker compose up -d
docker compose logs -f postgres   # wait until "ready to accept connections"
```

### Step 2 — Install Python deps & init schema

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the Alembic migration that creates the tables
alembic upgrade head
```

Open `alembic/versions/0001_init.py` and notice:
- It creates an `experiments` table and a `model_versions` table.
- `model_versions.experiment_id` is a FK to `experiments.id` with `ON DELETE CASCADE`.
- There is a **unique constraint** on `(experiment_id, version)` — this is what prevents "register version 5 twice".

### Step 3 — Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs>. Try:

1. `POST /experiments` with `{"name": "churn_v1", "owner": "alice"}`.
2. `POST /experiments/{id}/versions` twice with `{"version": "1.0.0", "metric_name": "auc", "metric_value": 0.83}`.
   - The second call must fail with **409 Conflict** — that's the unique constraint working.
3. `GET /experiments/{id}` — should return the experiment **with its versions in one query** (no N+1). Look at the server logs: there must be **one** SQL statement, not one per version.

### Step 4 — Look at the actual SQL Postgres runs

```bash
docker compose exec postgres psql -U mlops -d registry
# inside psql:
\dt
EXPLAIN ANALYZE SELECT * FROM model_versions WHERE experiment_id = 1;
\q
```

You'll see a `Seq Scan` because there's no index on `experiment_id` yet. **Add one** by editing the model and creating a new Alembic migration:

```bash
alembic revision --autogenerate -m "index on experiment_id"
alembic upgrade head
```

Re-run `EXPLAIN ANALYZE` — it should now use an `Index Scan`. That's the kind of question you'll be asked: *"how do you know an index is used?"*.

### Step 5 — Break and fix

Stop the API. Run two `psql` sessions side by side. In session A, `BEGIN; UPDATE experiments SET owner='bob' WHERE id=1;` (don't commit). In session B, run the same UPDATE — it blocks. `COMMIT` in A — B unblocks. Now you've *seen* row-level locking.

## Interview questions to rehearse

- Difference between `psycopg2` and `asyncpg`. Why does FastAPI benefit from async drivers?
- What is the default isolation level in Postgres? What does it *not* protect you from?
- Your endpoint becomes slow with 10 M rows in `model_versions`. What's your debugging order? (Answer outline: `EXPLAIN ANALYZE` → check index → check stats `ANALYZE` → check query plan → consider partitioning.)
- How do you do a zero-downtime schema migration that adds a `NOT NULL` column?
  (Add nullable → backfill → set NOT NULL → deploy. Never the other way.)
- `SELECT FOR UPDATE` vs optimistic locking with a `version` column — when to pick which?

## References

- PostgreSQL docs — Transaction Isolation: <https://www.postgresql.org/docs/current/transaction-iso.html>
- PostgreSQL docs — Indexes: <https://www.postgresql.org/docs/current/indexes.html>
- SQLAlchemy 2.0 async tutorial: <https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html>
- Alembic tutorial: <https://alembic.sqlalchemy.org/en/latest/tutorial.html>
- Use The Index, Luke! (the best free DB-indexing book): <https://use-the-index-luke.com/>
- "Postgres for everything": <https://www.amazingcto.com/postgres-for-everything/>
