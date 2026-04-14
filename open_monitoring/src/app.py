"""Periodic OpenBB monitor loop with Telegram alerts."""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
from concurrent.futures import ThreadPoolExecutor
import json
from collections.abc import Iterator
from typing import Callable
from datetime import datetime

from src.alert import AlertInfo, AlertInput
from src.alert_registry import registry
from src.telegram_controller import TelegramController
from src.persistent_data_controller import PersistentDataController
from src.app_config import app_config
from src.alerts import *

THREAD_POOL_SIZE = 4
THREAD_POOL_EXEC = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
)
_file_handler = logging.handlers.RotatingFileHandler(
    app_config.log_file_path,
    maxBytes=app_config.log_file_max_bytes,
    backupCount=0,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
logging.getLogger("src").addHandler(_file_handler)
logging.getLogger("__main__").addHandler(_file_handler)

log = logging.getLogger(__name__)

data_controller = PersistentDataController.with_sqlite(
    alerts_db_path=app_config.alerts_db_file_path,
    alert_buttons_db_path=app_config.alert_buttons_file_path,
    alerts_table_name="alerts",
    alert_button_table_name="alert_buttons",
)
telegram = TelegramController(
    alert_registry=registry,
    persistent_data_controller=data_controller,
    chat_id=app_config.chat_id,
    bot_token=app_config.bot_token,
    dry_run=app_config.dry_run,
)


async def load_alert_configs(default_alerts_path: str = app_config.default_alerts_path) -> list[AlertInfo]:
    db_alerts = await data_controller.get_alerts()
    if db_alerts:
        return db_alerts

    with open(default_alerts_path) as f:
        raw = json.load(f)
    return [AlertInfo(alert_input=AlertInput(**m)) for m in raw["alerts"]]


async def match_alerts(alert_configs: list[AlertInfo]) -> list[tuple[AlertInfo, Callable | None]]:
    alerts = [
        (alert_config, registry.get_alert_fn(alert_config.alert_input.fn))
        for alert_config in alert_configs
    ]
    for alert_config, alert_fn in alerts:
        if alert_fn is None:
            log.warning("Alert function not found for %s", alert_config.alert_input.fn)
            continue
    return [(alert_config, alert_fn) for alert_config, alert_fn in alerts if alert_fn is not None]


async def update_alert(alerts: list[AlertInfo], triggered_ids: list[str]) -> None:
    trigger_timestamp_sec = int(datetime.now(app_config.zone_info).timestamp())
    triggered_alerts = [alert for alert in alerts if alert.alert_input.id in triggered_ids]
    alerts_to_update = []
    for alert in triggered_alerts:
        if alert.alert_input.is_single_shot:
            await data_controller.remove_alert(alert.alert_input.id)
            continue

        alert.last_trigger_timestamp_sec = trigger_timestamp_sec
        alerts_to_update.append(alert)

    if alerts_to_update:
        await data_controller.update_alerts(alerts_to_update)


async def tick(
    alerts: Iterator[tuple[AlertInfo, Callable | None]],
    telegram: TelegramController,
) -> None:
    event_loop = asyncio.get_running_loop()

    alerts_to_invoke = [
        (alert_info, alert_fn)
        for alert_info, alert_fn in alerts
        if (
            alert_info.last_trigger_timestamp_sec is None
            or (alert_info.last_trigger_timestamp_sec + alert_info.alert_input.throttle_sec) <= int(datetime.now(app_config.zone_info).timestamp())
        )
    ]
    try:
        futures = [
            event_loop.run_in_executor(
                THREAD_POOL_EXEC,
                alert_fn,
                alert_config.alert_input,
            )
            for alert_config, alert_fn in alerts_to_invoke
        ]
        outputs = await asyncio.gather(*futures)
        alerts_to_send = [output for output in outputs if output.alert_message is not None]

        if alerts_to_send:
            log.info("Alerts to be sent: %d", len(alerts_to_send))
            await telegram.send_many(alerts_to_send)
            log.info("Updading alerts trigger throttles: %d", len(alerts_to_send))
            await update_alert(
                list[AlertInfo](map(lambda x: x[0], alerts_to_invoke)),
                [output.alert_id for output in alerts_to_send],
            )
    except Exception as e:
        log.exception("Monitoring failed with error: %s", e)


async def main_async() -> None:
    await data_controller.init()
    log.info("Run loop polling every %ds", app_config.runloop_interval_sec)

    async def pull() -> None:
        while True:
            alerts = await match_alerts(await load_alert_configs())
            await tick(alerts, telegram)
            await asyncio.sleep(app_config.runloop_interval_sec)

    await telegram.run(pull)


if __name__ == "__main__":
    asyncio.run(main_async())
