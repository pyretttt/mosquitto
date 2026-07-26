"""
records-api — FastAPI backend for the db_management lab.

Serves rows from Postgres. Bootstrap seeds a tiny `records` table on startup
when empty so Ingress / NetworkPolicy / backup tasks have something to observe.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from fastapi import FastAPI, HTTPException
from psycopg.rows import dict_row

app = FastAPI(title="records-api", version="0.1.0")


def _dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "records-pg-postgresql")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "records")
    user = os.getenv("POSTGRES_USER", "records")
    password = os.getenv("POSTGRES_PASSWORD", "")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is required")
    return f"host={host} port={port} dbname={db} user={user} password={password}"


@contextmanager
def _conn() -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(_dsn(), row_factory=dict_row) as conn:
        yield conn


def _ensure_schema_and_seed() -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT ''
                )
                """
            )
            cur.execute("SELECT COUNT(*) AS n FROM records")
            row = cur.fetchone()
            n = int(row["n"]) if row else 0
            if n == 0:
                cur.executemany(
                    "INSERT INTO records (name, note) VALUES (%s, %s)",
                    [
                        ("alpha", "seeded at startup"),
                        ("bravo", "use /records to list"),
                        ("charlie", "backup/restore should keep these"),
                    ],
                )
        conn.commit()


@app.on_event("startup")
def on_startup() -> None:
    # TODO(you): prefer an init Job / migration tool in production — TASKS.md §1
    try:
        _ensure_schema_and_seed()
    except Exception as exc:  # noqa: BLE001 — surface in /health for lab debugging
        app.state.db_error = str(exc)
    else:
        app.state.db_error = None


@app.get("/health")
def health() -> dict[str, object]:
    err = getattr(app.state, "db_error", None)
    if err:
        raise HTTPException(status_code=503, detail={"status": "db_error", "error": err})
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail={"status": "db_unreachable", "error": str(exc)}
        ) from exc
    return {"status": "ok"}


@app.get("/records")
def list_records() -> dict[str, object]:
    """Return all rows — the payload Ingress TLS and backup labs exercise."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, name, note FROM records ORDER BY id")
                rows = list(cur.fetchall())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"count": len(rows), "records": rows}


@app.post("/records")
def create_record(payload: dict[str, str]) -> dict[str, object]:
    name = (payload.get("name") or "").strip()
    note = (payload.get("note") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO records (name, note) VALUES (%s, %s) RETURNING id, name, note",
                (name, note),
            )
            row = cur.fetchone()
        conn.commit()
    return {"record": row}
