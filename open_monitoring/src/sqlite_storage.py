import asyncio
import sqlite3

import aiosqlite

from src.alert import AlertInput
from src.storage import Storage


class SQLiteStorage(Storage):
    def __init__(
        self,
        db_path: str,
        table_name: str = "alerts",
    ) -> None:
        self._db_path = db_path
        self._table_name = table_name


    async def init(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id   TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                )
            """)
            await db.commit()


    async def add(self, item: AlertInput) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                f"INSERT OR REPLACE INTO {self._table_name} (id, data) VALUES (?, ?)",
                (item.id, item.model_dump_json()),
            )
            await db.commit()


    async def get_all(self) -> list[AlertInput]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(f"SELECT data FROM {self._table_name}") as cursor:
                rows = await cursor.fetchall()
        return [AlertInput.model_validate_json(row[0]) for row in rows]


    async def remove(self, alert_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                f"DELETE FROM {self._table_name} WHERE id = ?", (alert_id,)
            )
            await db.commit()
            return cursor.rowcount > 0


    async def get(self, alert_id: str) -> AlertInput | None:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"SELECT data FROM {self._table_name} WHERE id = ?", (alert_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return AlertInput.model_validate_json(row[0])


    async def backup(self, dest_path: str) -> bool:
        def do_backup() -> None:
            src = sqlite3.connect(self._db_path)
            dst = sqlite3.connect(dest_path)
            try:
                # TODO: Add logging
                src.backup(dst)
                return True
            except Exception:
                return False
            finally:
                src.close()
                dst.close()
        return await asyncio.to_thread(do_backup)
