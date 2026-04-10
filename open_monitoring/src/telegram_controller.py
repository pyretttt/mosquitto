"""Telegram Bot API wrapper for sending monitoring alerts."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
import logging
from typing import Any

from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update

from src.alert import AlertMessage
from src.persistent_data_controller import PersistentDataController
from src.alert_registry import AlertRegistry


class Deps:
    alert_registry = "alert_registry"
    persistent_data_controller = "persistent_data_controller"


log = logging.getLogger(__name__)


class Commands:
    @staticmethod
    async def logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Sends explanation on how to use the bot."""
        await update.message.reply_text("Test")


class TelegramController:
    """Sends alert messages and handles bot commands via the Bot API."""

    def __init__(
        self,
        alert_registry: AlertRegistry,
        persistent_data_controller: PersistentDataController,
        chat_id: str,
        bot_token: str,
        dry_run: bool = False,

    ) -> None:
        self.alert_registry = alert_registry
        self.persistent_data_controller = persistent_data_controller
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.dry_run = dry_run
        self.application = Application.builder().token(bot_token).build()
        self.application.add_handler(CommandHandler("logs", Commands.logs))

        self.application.bot_data[Deps.alert_registry] = alert_registry
        self.application.bot_data[Deps.persistent_data_controller] = persistent_data_controller


    async def send(self, alert: AlertMessage) -> None:
        if not self.dry_run:
            await self.application.bot.send_message(
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
        async with self.application:
            await self.application.start()
            await self.application.updater.start_polling()

            await pull_fn()
            await self.application.updater.stop()
            await self.application.stop()
