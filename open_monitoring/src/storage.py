from abc import ABC, abstractmethod

from src.alert import AlertInput


class Storage(ABC):
    @abstractmethod
    async def init(self) -> None: ...

    @abstractmethod
    async def add(self, item: AlertInput) -> None: ...

    @abstractmethod
    async def get_all(self) -> list[AlertInput]: ...

    @abstractmethod
    async def remove(self, alert_id: str) -> bool: ...

    @abstractmethod
    async def get(self, alert_id: str) -> AlertInput | None: ...

    @abstractmethod
    async def backup(self, dest_path: str) -> bool: ...

    @abstractmethod
    async def restore(self, src_path: str) -> bool: ...