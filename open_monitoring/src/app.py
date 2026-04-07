"""Periodic OpenBB monitor loop with Telegram alerts."""

from __future__ import annotations

import asyncio
import logging
import os

import yaml

from src.data_controller import DataController
from src.monitor import Monitoring
from src.openbb_data_controller import OpenBBDataController
from src.telegram_controller import TelegramController

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
    controller: DataController,
    telegram: TelegramController,
) -> None:
    try:
        notifications = await controller.pull_and_get_notifications(monitors)
        if notifications:
            log.info("Notifications to be sent: %d", len(notifications))
            await telegram.send_many(notifications)
    except Exception as e:
        log.exception("Monitoring failed with error: %s", e)


async def main_async(period_sec: int = 60) -> None:
    obb_monitors = load_monitors()
    telegram = TelegramController(
        chat_id=os.environ["CHAT_ID"],
        bot_token=os.environ["BOT_TOKEN"],
        is_dry_run=os.environ["DRY_RUN"] == "true",
    )
    log.info("Loaded %d monitor(s), polling every %ds", len(obb_monitors), period_sec)

    controllers = {
        "obb": OpenBBDataController(),
    }
    while True:
        await tick(obb_monitors, controllers["obb"], telegram)
        await asyncio.sleep(period_sec)


if __name__ == "__main__":
    asyncio.run(main_async())
