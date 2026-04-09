import aiosqlite

from src.alert import AlertInput
from src.storage import Storage


class SQLiteStorage(Storage):
    def __init__(self, db_path: str = "alerts.db") -> None:
        self._db_path = db_path

    async def init(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS api_inputs (
                    id   TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                )
            """)
            await db.commit()

    async def add(self, item: AlertInput) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO api_inputs (id, data) VALUES (?, ?)",
                (item.id, item.model_dump_json()),
            )
            await db.commit()

    async def get_all(self) -> list[AlertInput]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT data FROM api_inputs") as cursor:
                rows = await cursor.fetchall()
        return [AlertInput.model_validate_json(row[0]) for row in rows]

    async def remove(self, alert_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM api_inputs WHERE id = ?", (alert_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get(self, alert_id: str) -> AlertInput | None:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT data FROM api_inputs WHERE id = ?", (alert_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return AlertInput.model_validate_json(row[0])
