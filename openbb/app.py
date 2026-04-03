"""Periodic OpenBB equity quote checks with Telegram alerts (async loop)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any
import yaml

from openbb import obb
from telegram import Bot


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


_last_alert_monotonic: float | None = None


async def main_async() -> None:
    with open("cfg.yaml", "r") as f:
        cfg = yaml.unsafe_load(f)

    log.info("Starting loop")

    # if cfg.dry_run:
    #     while True:
    #         try:
    #             await tick(None, cfg)
    #         except Exception:
    #             log.exception("tick failed")
    #         await asyncio.sleep(cfg.period_sec)

    # async with Bot(cfg.bot_token) as bot:
    #     while True:
    #         try:
    #             await tick(bot, cfg)
    #         except Exception:
    #             log.exception("tick failed")
    #         await asyncio.sleep(cfg.period_sec)


if __name__ == "__main__":
    asyncio.run(main_async())
