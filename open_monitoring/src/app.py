"""Periodic OpenBB monitor loop with Telegram alerts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
import time

import json

from src.data_controller import DataController
from src.monitor import Monitoring
from src.openbb_data_controller import OpenBBDataController, OpenBBContext
from src.telegram_controller import TelegramController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

@dataclass
class AppConfig:
    chat_id: str = os.environ["CHAT_ID"]
    bot_token: str = os.environ["BOT_TOKEN"]
    dry_run: bool = os.environ["DRY_RUN"] == "true"

app_config = AppConfig()


def load_obb_monitors(path: str = "configs/openbb_cfg.json") -> dict[str, Monitoring]:
    with open(path) as f:
        raw = json.load(f)
    obb_monitors = [Monitoring(**m) for m in raw["monitors"]]
    return {monitor.id: monitor for monitor in obb_monitors}


async def openbb_tick(
    monitors: dict[str, Monitoring],
    controller: OpenBBDataController,
    telegram: TelegramController,
) -> None:
    try:
        notifications = await controller.pull_and_get_notifications(
            ctx=OpenBBContext(
                monitors=monitors,
                timestamp=time.time(),
                dry_run=app_config.dry_run
            )
        )
        if notifications:
            log.info("Notifications to be sent: %d", len(notifications))
            await telegram.send_many(notifications)
    except Exception as e:
        log.exception("Monitoring failed with error: %s", e)


async def main_async(period_sec: int = 60) -> None:


    print(app_config)
    obb_monitors = load_obb_monitors()
    telegram = TelegramController(
        chat_id=app_config.chat_id,
        bot_token=app_config.bot_token,
        dry_run=app_config.dry_run == "true",
    )
    log.info("Loaded %d monitor(s), polling every %ds", len(obb_monitors), period_sec)

    controllers = {
        "obb": OpenBBDataController(),
    }
    while True:
        await openbb_tick(obb_monitors, controllers["obb"], telegram)
        await asyncio.sleep(period_sec)


if __name__ == "__main__":
    asyncio.run(main_async())
