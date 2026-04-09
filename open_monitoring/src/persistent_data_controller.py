from src.alert import AlertInput
from src.sqlite_storage import SQLiteStorage
from src.storage import Storage


class PersistentDataController:
    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    @classmethod
    def with_sqlite(cls, db_path: str = "alerts.db") -> "PersistentDataController":
        return cls(SQLiteStorage(db_path))

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
