"""Telegram Bot API wrapper for sending monitoring alerts."""

from __future__ import annotations

import logging
import os

from telegram import Bot
from telegram.constants import ParseMode

from src.api import AlertMessage

log = logging.getLogger(__name__)


class TelegramController:
    """Sends alert messages to a Telegram chat via the Bot API.

    Reads ``BOT_TOKEN`` and ``CHAT_ID`` from environment variables.
    """

    def __init__(
        self,
        chat_id: str,
        bot_token: str,
        dry_run: bool = False,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._bot = Bot(token=self.bot_token)
        self.dry_run = dry_run

    async def send(self, alert: AlertMessage) -> None:
        if not self.dry_run:
            await self._bot.send_message(
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
