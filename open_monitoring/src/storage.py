from abc import ABC, abstractmethod

from src.alert import AlertInfo


class Storage(ABC):
    @abstractmethod
    async def init(self) -> None: ...

    @abstractmethod
    async def add(self, item: AlertInfo) -> None: ...

    @abstractmethod
    async def get_all(self) -> list[AlertInfo]: ...

    @abstractmethod
    async def remove(self, alert_id: str) -> bool: ...

    @abstractmethod
    async def get(self, alert_id: str) -> AlertInfo | None: ...

    @abstractmethod
    async def backup(self, dest_path: str) -> bool: ...

    @abstractmethod
    async def restore(self, src_path: str) -> bool: ...

    @abstractmethod
    async def update_alerts(self, alerts: list[AlertInfo]) -> None: ...