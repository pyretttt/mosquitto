"""Database engine and session factory.

Why async?
- FastAPI handles requests in an event loop. A blocking psycopg2 call would
  pin a thread per request. With asyncpg, the event loop can serve other
  requests while Postgres is computing.

Why a pool (and pool_size=10)?
- TCP + auth handshake to Postgres takes ~5 ms. Doing that per request is
  insane. The pool keeps N connections open; requests check one out and back.
- pool_size should be <= max_connections on the server (default 100 in PG).
  In production you usually put PgBouncer in front and use a smaller app-side
  pool.
"""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL = "postgresql+asyncpg://mlops:mlops@localhost:5432/registry"

engine = create_async_engine(
    DATABASE_URL,
    echo=True,           # logs every SQL — great for spotting N+1 in dev
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,  # validates conn before use; survives PG restarts
)

SessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a session per request."""
    async with SessionLocal() as session:
        yield session
