import secrets
import sqlite3
import time

import aiosqlite

from src.alert import AlertInput, AlertButton
from src.sqlite_storage import SQLiteStorage
from src.storage import Storage

_CALLBACK_TOKENS_TABLE = "callback_tokens"
_CALLBACK_TOKENS_MAX_ROWS = 1000


class PersistentDataController:
    def __init__(self, storage: Storage, db_path: str | None = None) -> None:
        self._storage = storage
        self._db_path = db_path

    @classmethod
    def with_sqlite(cls, db_path: str, table_name: str) -> "PersistentDataController":
        return cls(SQLiteStorage(db_path, table_name), db_path=db_path)

    async def init(self) -> None:
        await self._storage.init()
        if self._db_path is not None:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_CALLBACK_TOKENS_TABLE} (
                        id TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        created_at INTEGER NOT NULL
                    )
                    """
                )
                await db.commit()

    async def add_alert(self, item: AlertInput) -> None:
        await self._storage.add(item)

    async def get_alerts(self) -> list[AlertInput]:
        return await self._storage.get_all()

    async def remove_alert(self, alert_id: str) -> bool:
        return await self._storage.remove(alert_id)

    async def get_alert(self, alert_id: str) -> AlertInput | None:
        return await self._storage.get(alert_id)

    async def backup(self, dest_path: str) -> bool:
        return await self._storage.backup(dest_path)

    async def restore(self, src_path: str) -> bool:
        return await self._storage.restore(src_path)

    async def register_alert_button(self, button: AlertButton) -> str:
        if self._db_path is None:
            raise RuntimeError("register_callback_payload requires sqlite db_path")
        payload = button.model_dump_json()
        for _ in range(8):
            token_id = secrets.token_hex(8)
            try:
                async with aiosqlite.connect(self._db_path) as db:
                    await db.execute(
                        f"""
                        INSERT INTO {_CALLBACK_TOKENS_TABLE} (id, payload, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (token_id, payload, int(time.time())),
                    )
                    await db.execute(
                        f"""
                        DELETE FROM {_CALLBACK_TOKENS_TABLE} WHERE id IN (
                            SELECT id FROM {_CALLBACK_TOKENS_TABLE}
                            ORDER BY created_at ASC
                            LIMIT (
                                SELECT CASE WHEN cnt > ? THEN cnt - ? ELSE 0 END
                                FROM (SELECT COUNT(*) AS cnt FROM {_CALLBACK_TOKENS_TABLE})
                            )
                        )
                        """,
                        (_CALLBACK_TOKENS_MAX_ROWS, _CALLBACK_TOKENS_MAX_ROWS),
                    )
                    await db.commit()
                return token_id
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("Failed to allocate callback token after retries")

    async def get_alert_button(self, token: str) -> AlertButton:
        if self._db_path is None:
            return None
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"SELECT payload FROM {_CALLBACK_TOKENS_TABLE} WHERE id = ?",
                (token,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        try:
            return AlertButton.model_validate_json(row[0])
        except Exception:
            return None
