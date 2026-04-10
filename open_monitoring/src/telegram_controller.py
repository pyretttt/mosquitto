"""Telegram Bot API wrapper for sending monitoring alerts."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
import logging
from typing import Any
import tempfile
from typing import List

from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, filters
from telegram import Update
from pydantic import RootModel

from src.alert import AlertMessage
from src.persistent_data_controller import PersistentDataController
from src.alert_registry import Registry
from src.app_config import app_config


class Deps:
    alert_registry = "alert_registry"
    persistent_data_controller = "persistent_data_controller"


log = logging.getLogger(__name__)


class Commands:
    @staticmethod
    async def show_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Sends start message."""
        await update.message.reply_text("Preparing alerts list...")
        persistent_data_controller = context.bot_data[Deps.persistent_data_controller]
        all_alerts =await persistent_data_controller.get_alerts()
        class AlertList(RootModel):
            root: List[AlertMessage]
        alert_list = AlertList(root=all_alerts)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=True) as tmp:
            print("alert_list.model_dump_json(indent=2): ", alert_list.model_dump_json(indent=2))
            tmp.write(alert_list.model_dump_json(indent=2).encode("utf-8"))
            tmp.flush()
            await context.bot.send_document(
                chat_id=update.message.chat_id,
                document=open(tmp.name, "r", encoding="utf-8")
            )


    @staticmethod
    async def logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Sends logs file."""
        await update.message.reply_text("Preparing logs...")
        await context.bot.send_document(
            chat_id=update.message.chat_id,
            document=open(app_config.log_file_path, "r")
        )

    @staticmethod
    async def dump_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Sends logs file."""
        await update.message.reply_text("Preparing database file...")
        persistent_data_controller = context.bot_data[Deps.persistent_data_controller]
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            is_success = await persistent_data_controller.backup(tmp.name)
            if is_success:
                await context.bot.send_document(
                    chat_id=update.message.chat_id,
                    document=open(tmp.name, "rb")
                )
            else:
                await update.message.reply_text("Failed to send database file.")


class TelegramController:
    """Sends alert messages and handles bot commands via the Bot API."""

    def __init__(
        self,
        alert_registry: Registry,
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
        chat_filter = filters.Chat(chat_id=[int(chat_id)])
        self.application.add_handler(CommandHandler("logs", Commands.logs, filters=chat_filter))
        self.application.add_handler(CommandHandler("dump_db", Commands.dump_db, filters=chat_filter))
        self.application.add_handler(CommandHandler("alerts", Commands.show_alerts, filters=chat_filter))

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
