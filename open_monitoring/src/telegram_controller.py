"""Telegram Bot API wrapper for sending monitoring alerts."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
import logging
import re
from typing import Any
import tempfile
from typing import List
import asyncio

from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram import Update
from pydantic import RootModel

from src.alert import AlertInput, AlertMessage
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
        persistent_data_controller: PersistentDataController = context.bot_data[Deps.persistent_data_controller]
        all_alerts = await persistent_data_controller.get_alerts()

        class AlertList(RootModel):
            root: List[AlertMessage]

        alert_list = AlertList(root=all_alerts)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=True) as tmp:
            print("alert_list.model_dump_json(indent=2): ", alert_list.model_dump_json(indent=2))
            tmp.write(alert_list.model_dump_json(indent=2).encode("utf-8"))
            tmp.flush()
            await context.bot.send_document(
                chat_id=update.message.chat_id, document=open(tmp.name, "r", encoding="utf-8")
            )

    @staticmethod
    async def logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Sends logs file."""
        await update.message.reply_text("Preparing logs...")
        await context.bot.send_document(chat_id=update.message.chat_id, document=open(app_config.log_file_path, "r"))

    @staticmethod
    async def set_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Sends logs file."""
        document = update.message.document
        if document is None:
            await update.message.reply_text("Please send a SQLite database file as a document.")
            return

        await update.message.reply_text("Restoring database...")
        persistent_data_controller: PersistentDataController = context.bot_data[Deps.persistent_data_controller]
        with tempfile.NamedTemporaryFile(delete=True, suffix=".db") as tmp:
            tg_file = await context.bot.get_file(document.file_id)
            await tg_file.download_to_drive(tmp.name)
            is_success = await persistent_data_controller.restore(tmp.name)
            if is_success:
                await update.message.reply_text("Database restored successfully.")
            else:
                await update.message.reply_text("Failed to restore database.")

    @staticmethod
    async def dump_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Sends logs file."""
        await update.message.reply_text("Preparing database file...")
        persistent_data_controller: PersistentDataController = context.bot_data[Deps.persistent_data_controller]
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            is_success = await persistent_data_controller.backup(tmp.name)
            if is_success:
                await context.bot.send_document(chat_id=update.message.chat_id, document=open(tmp.name, "rb"))
            else:
                await update.message.reply_text("Failed to send database file.")

    @staticmethod
    async def remove_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Removes an alert."""
        try:
            alert_id = update.message.text.split(" ")[1]
            persistent_data_controller: PersistentDataController = context.bot_data[Deps.persistent_data_controller]
            is_success = await persistent_data_controller.remove_alert(alert_id)
            if is_success:
                await update.message.reply_text("Alert removed successfully.")
            else:
                await update.message.reply_text("Failed to remove alert.")
        except Exception:
            await update.message.reply_text("Failed to remove alert.")

    @staticmethod
    async def alert_desc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Sends alert description."""
        try:
            alert_id = update.message.text.split(" ")[1]
            persistent_data_controller: PersistentDataController = context.bot_data[Deps.persistent_data_controller]
            alert = await persistent_data_controller.get_alert(alert_id)
            if a := alert:
                text = "```\n" + a.model_dump_json(indent=2) + "\n```"
                await update.message.reply_text(text)
            else:
                await update.message.reply_text("Alert with this ID not found. 🆘")
        except Exception:
            await update.message.reply_text("Failed to find alert. Maybe wrong ID? 🆘")

    @staticmethod
    async def add_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Adds an alert from a JSON code block in the message."""
        try:
            text = update.message.text or ""
            match = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
            if not match:
                await update.message.reply_text("Please include a JSON code block with alert definition.")
                return

            raw_json = match.group(1).strip()
            alert_input = AlertInput.model_validate_json(raw_json)

            alert_registry: Registry = context.bot_data[Deps.alert_registry]
            if alert_input.fn not in alert_registry:
                await update.message.reply_text(f"Alert function '{alert_input.fn}' is not registered.")
                return

            try:
                await asyncio.run_in_executor(None, alert_registry[alert_input.fn], **alert_input.params)
            except Exception:
                await update.message.reply_text("Failed to execute alert function. Check your JSON format. 🆘")
                return

            persistent_data_controller: PersistentDataController = context.bot_data[Deps.persistent_data_controller]
            await persistent_data_controller.add_alert(alert_input)
            await update.message.reply_text(f"Alert '{alert_input.name}' added successfully. 🆘")
        except Exception:
            await update.message.reply_text("Failed to add alert. Check your JSON format. 🆘")


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
        self.application.add_handler(
            MessageHandler(
                chat_filter & filters.Document.ALL & filters.Caption(["/set_db"]),
                Commands.set_db,
            )
        )
        self.application.add_handler(
            MessageHandler(
                chat_filter & filters.Text(strings=["/remove_alert"]),
                Commands.remove_alert,
            )
        )
        self.application.add_handler(
            MessageHandler(
                chat_filter & filters.Text(strings=["/show_alert_desc"]),
                Commands.remove_alert,
            )
        )
        self.application.add_handler(
            MessageHandler(
                chat_filter & filters.Text(strings=["/new_alert"]),
                Commands.add_alert,
            )
        )

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
