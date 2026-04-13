import asyncio
import secrets
import sqlite3
import time

import aiosqlite

from src.alert import AlertButton, AlertInfo
from src.storage import Storage

_ALERT_BUTTON_MAX_ROWS = 1000


class SQLiteStorage(Storage):
    def __init__(
        self,
        alerts_db_path: str,
        alert_buttons_db_path: str,
        alerts_table_name: str,
        alert_button_table_name: str,
    ) -> None:
        self.alerts_db_path = alerts_db_path
        self.alert_buttons_db_path = alert_buttons_db_path
        self.alerts_table_name = alerts_table_name
        self.alert_button_table_name = alert_button_table_name

    async def init(self) -> None:
        async with aiosqlite.connect(self.alerts_db_path) as db:
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.alerts_table_name} (
                    id   TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                )
                """
            )
            await db.commit()
        async with aiosqlite.connect(self.alert_buttons_db_path) as db:
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.alert_button_table_name} (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            await db.commit()

    async def add(self, item: AlertInfo) -> None:
        async with aiosqlite.connect(self.alerts_db_path) as db:
            await db.execute(
                f"INSERT OR REPLACE INTO {self.alerts_table_name} (id, data) VALUES (?, ?)",
                (item.alert_input.id, item.model_dump_json()),
            )
            await db.commit()

    async def get_all(self) -> list[AlertInfo]:
        async with aiosqlite.connect(self.alerts_db_path) as db:
            async with db.execute(f"SELECT data FROM {self.alerts_table_name}") as cursor:
                rows = await cursor.fetchall()
        return [AlertInfo.model_validate_json(row[0]) for row in rows]

    async def remove(self, alert_id: str) -> bool:
        async with aiosqlite.connect(self.alerts_db_path) as db:
            cursor = await db.execute(f"DELETE FROM {self.alerts_table_name} WHERE id = ?", (alert_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def get(self, alert_id: str) -> AlertInfo | None:
        async with aiosqlite.connect(self.alerts_db_path) as db:
            async with db.execute(f"SELECT data FROM {self.alerts_table_name} WHERE id = ?", (alert_id,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return AlertInfo.model_validate_json(row[0])

    async def backup(self, dest_path: str) -> bool:
        def do_backup() -> None:
            src = sqlite3.connect(self.alerts_db_path)
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

    async def restore(self, src_path: str) -> bool:
        def do_restore() -> bool:
            src = sqlite3.connect(src_path)
            dst = sqlite3.connect(self.alerts_db_path)
            try:
                src.backup(dst)
                return True
            except Exception:
                return False
            finally:
                src.close()
                dst.close()

        return await asyncio.to_thread(do_restore)

    async def register_alert_button(self, button: AlertButton) -> str:
        payload = button.model_dump_json()
        for _ in range(8):
            token_id = secrets.token_hex(8)
            try:
                async with aiosqlite.connect(self.alert_buttons_db_path) as db:
                    await db.execute(
                        f"""
                        INSERT INTO {self.alert_button_table_name} (id, payload, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (token_id, payload, int(time.time())),
                    )
                    await db.execute(
                        f"""
                        DELETE FROM {self.alert_button_table_name} WHERE id IN (
                            SELECT id FROM {self.alert_button_table_name}
                            ORDER BY created_at ASC
                            LIMIT (
                                SELECT CASE WHEN cnt > ? THEN cnt - ? ELSE 0 END
                                FROM (SELECT COUNT(*) AS cnt FROM {self.alert_button_table_name})
                            )
                        )
                        """,
                        (_ALERT_BUTTON_MAX_ROWS, _ALERT_BUTTON_MAX_ROWS),
                    )
                    await db.commit()
                return token_id
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("Failed to allocate callback token after retries")

    async def get_alert_button(self, token: str) -> AlertButton | None:
        async with aiosqlite.connect(self.alert_buttons_db_path) as db:
            async with db.execute(
                f"SELECT payload FROM {self.alert_button_table_name} WHERE id = ?", (token,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        try:
            return AlertButton.model_validate_json(row[0])
        except Exception:
            return None
