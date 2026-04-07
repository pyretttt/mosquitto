from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor


from openbb import obb, OBBject
from asyncio import AbstractEventLoop, get_running_loop
from pydantic import BaseModel

from src.utils import resolve_method
from src.condition import ConditionEvaluationResult, evaluate_condition
from src.monitor import Monitoring, Pull
from src.data_controller import DataController

log = logging.getLogger(__name__)

ALLOWED_ROOTS: frozenset[str] = frozenset[str]({
    "commodity", "crypto", "currency", "derivatives", "economy",
    "equity", "etf", "fixedincome", "index", "news", "regulators",
})
THREAD_POOL_SIZE = 4
THREAD_POOL_EXEC = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)


class OpenBBContext(BaseModel):
    """Context of openbb data controller."""
    monitors: dict[str, Monitoring]
    timestamp: float
    dry_run: bool


class OpenBBData(BaseModel):
    """Data from openbb data controller."""
    openbb_data: dict[str, OBBject]


def pull_data(pull: Pull) -> OBBject:
    """Call an OpenBB endpoint described by *pull* and return the first result as a dict."""
    fn = resolve_method(obb, pull.method, ALLOWED_ROOTS)
    data = fn(**pull.params)
    if data is None:
        raise ValueError(f"No results from {pull.method}")
    return data


def format_template(template: str, data: dict) -> str:
    """Replace ``${data.field}`` placeholders with values from *data*.

    Supports nested paths like ``${data.nested.field}``.
    """
    def _resolve(match: re.Match) -> str:
        cur: object = {"data": data}
        for part in match.group(1).split("."):
            cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
        return str(cur)

    return re.sub(r"\$\{([^}]+)\}", _resolve, template)


class OpenBBDataController(DataController[OpenBBContext, OpenBBData]):

    def __init__(self):
        self.timeouts_until = dict[str, float]()


    @property
    def runloop(self) -> AbstractEventLoop:
        return get_running_loop()


    async def pull(self, ctx: OpenBBContext) -> OpenBBData:
        openbb_data = OpenBBData(openbb_data=dict[str, OBBject]())

        items_to_pull = {
            id: monitor for id, monitor in ctx.monitors.items()
            if self.timeouts_until.get(id, 0) > time.time()
        }
        for id, monitor in items_to_pull.items():
            try:
                data = await self.runloop.run_in_executor(THREAD_POOL_EXEC, pull_data, monitor.pull)
                openbb_data.openbb_data[id] = data

            except Exception as e:
                log.error(f"Error pulling data: {e}")

        return openbb_data


    def evaluate_conditions(self, data: OpenBBData, ctx: OpenBBContext) -> dict[str, ConditionEvaluationResult]:
        conditions = dict[str, ConditionEvaluationResult]()
        for id, data in data.openbb_data.items():
            try:
                conditions[id] = evaluate_condition(data=data, expression=ctx.monitors[id].condition)
            except Exception as e:
                log.error(f"Error evaluating condition: {e}")

        return conditions


    def notify(
        self,
        data: OpenBBData,
        ctx: OpenBBContext,
        conditions: dict[str, ConditionEvaluationResult]
    ) -> bool:
        for id, condition in conditions.items():
            if not condition.triggered:
                continue
            monitor = ctx.monitors[id]
            data = data.openbb_data[id]
            self.timeouts_until[id] = time.time() + monitor.notify.telegram.min_interval_sec
            monitor.notify.telegram.template.format_map(
                **data["results"]
            )
