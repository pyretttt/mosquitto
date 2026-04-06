import openbb
from pydantic import BaseModel

from monitor import Monitor
from aggregator import DataController


class OpenBBConfig(BaseModel):
    """Configures pull and notify behavior."""

    monitors: list[Monitor]


class OpenBBContext:
    """Context for the OpenBB aggregator."""

    timestamp: float
    dry_run: bool


class OpenBBData:
    pass


class OpenBBDataController(DataController[OpenBBContext, OpenBBData, OpenBBConfig]):
    def __init__(self):
        self.client = openbb.Client()

    def pull(self, ctx: OpenBBContext) -> OpenBBData:
        pass
