"""Periodic OpenBB monitor loop with Telegram alerts."""

from __future__ import annotations

import asyncio
import logging

import yaml

from src.monitor import Monitoring
from src.openbb_data_controller import run_monitoring
from src.openbb_data_controller import OpenBBDataController
from src.data_controller import DataController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def load_monitors(path: str = "openbb_cfg.yaml") -> list[Monitoring]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [Monitoring(**m) for m in raw["monitors"]]


async def tick(
    monitors: list[Monitoring],
    controller: DataController
) -> None:
    try:
        notifications = await controller.pull_and_get_notifications(monitors)
        if notifications:
            log.info(f"Notifications to be sent: {len(notifications)}")
            # TODO: send via telegram bot
    except Exception as e:
        log.exception(f"Monitoring failed with error: {e}")


async def main_async(period_sec: int = 60) -> None:
    obb_monitors = load_monitors()
    log.info("Loaded %d monitor(s), polling every %ds", len(obb_monitors), period_sec)

    controllers = {
        "obb": OpenBBDataController(),
    }
    while True:
        await tick(obb_monitors, controllers["obb"])
        await asyncio.sleep(period_sec)


if __name__ == "__main__":
    asyncio.run(main_async())
