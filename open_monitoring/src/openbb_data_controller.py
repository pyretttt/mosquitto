from __future__ import annotations

import functools
import logging
import re

from openbb import obb, OBBject
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


class OpenBBContext(BaseModel):
    """Context of openbb data controller."""
    monitors: list[Monitoring]
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


def run_monitoring(monitor: Monitoring) -> tuple[ConditionEvaluationResult, str | None]:
    """Pull → evaluate → format.  Returns (condition_result, formatted_message | None)."""
    data = pull_data(monitor.pull)
    result = evaluate_condition(data, monitor.condition)
    message = format_template(monitor.notify.telegram.template, data) if result.triggered else None
    return result, message


class OpenBBDataController(DataController[OpenBBContext, OpenBBData]):
    def pull(self, monitor: Monitoring) -> dict:
        return pull_data(monitor.pull)

    def format_message(self, monitor: Monitoring, data: dict) -> str:
        return format_template(monitor.notify.telegram.template, data)

    def evaluate(self, data: dict, expression: str) -> ConditionEvaluationResult:
        return evaluate_condition(data, expression)


    def pull(self, ctx: OpenBBContext) -> OpenBBData:
        try:
            for monitor in ctx.monitors:
                try:
                return self.pull(monitor)
        except Exception as e:
            log.error(f"Error pulling data: {e}")
            return None


    def find_conditions(self, data: OpenBBData, ctx: OpenBBContext) -> list[ConditionEvaluationResult]:
        pass


    def notify(self, ctx: OpenBBContext, conditions: list[ConditionEvaluationResult]) -> bool:
        pass