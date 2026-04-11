from src.alert import AlertButton, AlertInput
from src.sqlite_storage import SQLiteStorage
from src.storage import Storage


class PersistentDataController:
    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    @classmethod
    def with_sqlite(
        cls,
        alerts_db_path: str,
        alert_buttons_db_path: str,
        alerts_table_name: str,
        alert_button_table_name: str,
    ) -> "PersistentDataController":
        return cls(
            SQLiteStorage(
                alerts_db_path=alerts_db_path,
                alert_buttons_db_path=alert_buttons_db_path,
                alerts_table_name=alerts_table_name,
                alert_button_table_name=alert_button_table_name,
            )
        )

    async def init(self) -> None:
        await self._storage.init()

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
        return await self._storage.register_alert_button(button)

    async def get_alert_button(self, token: str) -> AlertButton | None:
        return await self._storage.get_alert_button(token)
