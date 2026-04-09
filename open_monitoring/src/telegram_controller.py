"""Telegram Bot API wrapper for sending monitoring alerts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
import logging
import time
from typing import Any

from telegram.constants import ParseMode
from telegram.ext import Application

from src.alert import AlertMessage

log = logging.getLogger(__name__)


class TelegramController:
    """Sends alert messages and handles bot commands via the Bot API."""

    def __init__(
        self,
        chat_id: str,
        bot_token: str,
        dry_run: bool = False,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.dry_run = dry_run
        self._application = Application.builder().token(bot_token).build()


    async def send(self, alert: AlertMessage) -> None:
        if not self.dry_run:
            await self._application.bot.send_message(
                chat_id=self.chat_id,
                text=alert.format(),
                parse_mode=ParseMode.MARKDOWN,
            )
        log.info("Sent alert %s", alert.name)


    async def send_many(self, alerts: list[AlertMessage]) -> None:
        for alert in alerts:
            try:
                await self.send(alert)
            except Exception:
                log.exception("Failed to send alert %s", alert.name)


    async def run(
        self,
        pull_fn: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Run the bot and call monitor_fn at most once per interval_sec."""
        async with self._application:
            await self._application.start()
            await self._application.updater.start_polling()

            await pull_fn()
            await self._application.updater.stop()
            await self._application.stop()
