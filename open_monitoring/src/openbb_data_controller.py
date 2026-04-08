from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from openbb import obb
import asyncio
from asyncio import get_running_loop
from pydantic import BaseModel

from src.utils import resolve_method, resolve_attr
from src.condition import ConditionEvaluationResult, evaluate_json_condition
from src.monitor import Monitoring
from src.data_controller import DataController
from src.notification import Notification
from src.utils import Safedict

log = logging.getLogger(__name__)

ALLOWED_METHOD_ROOTS: frozenset[str] = frozenset[str]({
    "commodity", "crypto", "currency", "derivatives", "economy",
    "equity", "etf", "fixedincome", "index", "news", "regulators",
})
ALLOWED_DATA_ROOTS: frozenset[str] = frozenset[str]({
    "results",
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
    openbb_data: dict[str, dict]


def pull_data(monitor: Monitoring) -> tuple[Monitoring, dict | None]:
    """Call an OpenBB endpoint described by *pull* and return the first result as a dict."""
    try:
        fn = resolve_method(obb, monitor.pull.method, ALLOWED_METHOD_ROOTS)
        data = fn(**monitor.pull.params)
        if data is None:
            raise ValueError(f"No results from {monitor.pull.method}")
        return monitor, data
    except Exception:
        log.exception(f"Error pulling data")
        return monitor, None


def transform_data(data: object, transforms: list[str]):
    current_data = data
    for transform in transforms:
        match transform:
            case "to_dict":
                current_data = current_data.to_dict()
            case "last":
                current_data = current_data[-1]
            case "model_dump":
                current_data = current_data.model_dump()
            case _:
                raise ValueError(f"Unsupported data type: {type}")
    return current_data


class OpenBBDataController(DataController[OpenBBContext, OpenBBData]):

    def __init__(self):
        self.timeouts_until: dict[str, float] = dict[str, float]()


    async def pull(self, ctx: OpenBBContext) -> OpenBBData:
        openbb_data = OpenBBData(openbb_data=dict[str, dict]())
        items_to_pull = {
            id: monitor for id, monitor in ctx.monitors.items()
            if self.timeouts_until.get(id, 0) <= time.time()
        }
        loop = get_running_loop()
        tasks = [
            loop.run_in_executor(THREAD_POOL_EXEC, pull_data, monitor)
            for monitor in items_to_pull.values()
        ]
        results: list[tuple[Monitoring, dict | None]] = await asyncio.gather(*tasks)
        for monitor, data in results:
            if data is None:
                log.error(f"Error pulling data: {monitor}")
                continue
            openbb_data.openbb_data[monitor.id] = data

        return openbb_data


    def evaluate_conditions(self, data: OpenBBData, ctx: OpenBBContext) -> dict[str, ConditionEvaluationResult]:
        eval_conditions = dict[str, ConditionEvaluationResult]()
        for id, data in data.openbb_data.items():
            try:
                condition = ctx.monitors[id].condition
                evaluation_data = resolve_attr(
                    root=data,
                    dotted_path=condition.data_key,
                    allowed_roots=ALLOWED_DATA_ROOTS
                )
                eval_conditions[id] = evaluate_json_condition(
                    labeled_data=(condition.data_key, transform_data(data=evaluation_data, transforms=condition.data_transforms)),
                    expression=condition.expression
                )
            except Exception as e:
                log.exception("Error evaluating condition")

        return eval_conditions


    def get_notifications(
        self,
        data: OpenBBData,
        ctx: OpenBBContext,
        conditions: dict[str, ConditionEvaluationResult]
    ) -> list[Notification]:
        notifications = list[Notification]()
        for id, condition in conditions.items():
            if not condition.triggered:
                continue
            monitor = ctx.monitors[id]
            print(data.openbb_data[id])
            item_data = resolve_attr(
                root=data.openbb_data[id],
                dotted_path=monitor.notify.data_key,
                allowed_roots=ALLOWED_DATA_ROOTS
            )
            item_data = transform_data(data=item_data, transforms=monitor.notify.data_transforms)
            self.timeouts_until[id] = time.time() + monitor.notify.min_interval_sec

            alert_message = monitor.notify.template.format_map(
                Safedict(item_data)
            )
            notifications.append(Notification(monitor=monitor, alert_message=alert_message))
            log.info(f"Alert: {alert_message}")
        return notifications