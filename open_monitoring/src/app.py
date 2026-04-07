"""Periodic OpenBB monitor loop with Telegram alerts."""

from __future__ import annotations

import asyncio
import logging

import yaml

from src.monitor import Monitoring
from src.openbb_data_controller import run_monitoring
from src.openbb_data_controller import OpenBBDataController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def load_monitors(path: str = "openbb_cfg.yaml") -> list[Monitoring]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [Monitoring(**m) for m in raw["monitors"]]


async def tick(monitors: list[Monitoring]) -> None:
    root_to_controller = {
        "obb": OpenBBDataController(),
    }

    for mon in monitors:
        try:
            data_controller = root_to_controller[mon.pull.root]
            eval_result, message = data_controller.run(mon)
            if eval_result.triggered:
                log.info("[%s] TRIGGERED — %s", mon.name, message)
                # TODO: send via telegram bot
            else:
                log.debug("[%s] condition not met", mon.name)
        except Exception:
            log.exception("[%s] monitor failed", mon.name)


async def main_async(period_sec: int = 60) -> None:
    monitors = load_monitors()
    log.info("Loaded %d monitor(s), polling every %ds", len(monitors), period_sec)

    while True:
        await tick(monitors)
        await asyncio.sleep(period_sec)


if __name__ == "__main__":
    asyncio.run(main_async())
