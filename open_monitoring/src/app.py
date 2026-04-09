"""Periodic OpenBB monitor loop with Telegram alerts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import json
from collections.abc import Iterator
from typing import Callable
import time

from src.alert import AlertInput
from src.alert_registry import registry
from src.telegram_controller import TelegramController
from src.persistent_data_controller import PersistentDataController
import src.apis


THREAD_POOL_SIZE = 4
THREAD_POOL_EXEC = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)


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
    runloop_interval_sec: int = int(os.environ.get("RUNLOOP_INTERVAL_SEC", "60"))


app_config = AppConfig()
data_controller = PersistentDataController.with_sqlite(db_path="runtime/alerts.db", table_name="alerts")


async def load_alert_configs(default_alerts_path: str = "configs/default_alerts.json") -> list[AlertInput]:
    db_alerts = await data_controller.get_alerts()
    if db_alerts:
        return db_alerts

    with open(default_alerts_path) as f:
        raw = json.load(f)
    alerts = [AlertInput(**m) for m in raw["alerts"]]
    return alerts


async def match_alerts(alert_configs: list[AlertInput]) -> list[tuple[AlertInput, Callable | None]]:
    alerts = [(alert_config, registry.get(alert_config.fn)) for alert_config in alert_configs]
    for alert_config, alert_fn in alerts:
        if alert_fn is None:
            log.warning("Alert function not found for %s", alert_config.fn)
            continue
    return [(alert_config, alert_fn) for alert_config, alert_fn in alerts if alert_fn is not None]


async def tick(
    alerts: Iterator[tuple[AlertInput, Callable | None]],
    telegram: TelegramController,
) -> None:
    event_loop = asyncio.get_running_loop()
    try:
        futures = [
            event_loop.run_in_executor(
                THREAD_POOL_EXEC,
                alert_fn,
                alert_config
            )
            for alert_config, alert_fn in alerts
        ]
        outputs = await asyncio.gather(*futures)
        alerts_to_send = [output.alert_message for output in outputs if output.alert_message is not None]
        if alerts_to_send:
            log.info("Alerts to be sent: %d", len(alerts_to_send))
            await telegram.send_many(alerts_to_send)
    except Exception as e:
        log.exception("Monitoring failed with error: %s", e)


async def main_async() -> None:
    telegram = TelegramController(
        chat_id=app_config.chat_id,
        bot_token=app_config.bot_token,
        dry_run=app_config.dry_run,
    )

    log.info("Run loop polling every %ds", app_config.runloop_interval_sec)

    async def pull() -> None:
        while True:
            alerts = await match_alerts(await load_alert_configs())
            await tick(alerts, telegram)
            await asyncio.sleep(app_config.runloop_interval_sec)

    await telegram.run(pull)


if __name__ == "__main__":
    asyncio.run(main_async())
