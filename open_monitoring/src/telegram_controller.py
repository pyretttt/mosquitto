"""Telegram Bot API wrapper for sending monitoring alerts."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
import asyncio
import logging
import re
from typing import Any
import tempfile
from typing import List
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters
from pydantic import RootModel

from src.alert import AlertButton, AlertInput, AlertMessage, AlertOutput, AlertInfo
from src.persistent_data_controller import PersistentDataController
from src.alert_registry import Registry
from src.app_config import app_config


class Deps:
    alert_registry = "alert_registry"
    persistent_data_controller = "persistent_data_controller"


log = logging.getLogger(__name__)


async def inline_keyboard_from_buttons(
    buttons: list[AlertButton],
    persistent_data_controller: PersistentDataController,
) -> InlineKeyboardMarkup | None:
    """Build an inline keyboard; callback_data is a short token mapped in SQLite."""
    rows: list[list[InlineKeyboardButton]] = []
    for b in buttons:
        try:
            token = await persistent_data_controller.register_alert_button(b)
        except Exception:
            log.exception("Skipping button %r: failed to register callback payload", b.name)
            continue
        rows.append([InlineKeyboardButton(b.name, callback_data=token)])
    return InlineKeyboardMarkup(rows) if rows else None


class Commands:
    @staticmethod
    async def alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    async def functionality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Lists all registered alert functions with their descriptions."""
        alert_registry: Registry = context.bot_data[Deps.alert_registry]
        lines: list[str] = []
        for name, fn in alert_registry.alert_map.items():
            escaped_name = name.replace("_", "\\_")
            doc = (fn.__doc__ or "No description").strip().replace("_", "\\_")
            lines.append(f"*{escaped_name}*\n{doc}")
        text = "\n\n".join(lines) if lines else "No functions registered."
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

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
    async def show_alert_desc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    async def new_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Adds an alert from a JSON code block in the message."""
        try:
            text = update.message.text or ""
            entities = update.message.entities
            entity = next(entity for entity in entities if entity.type == "pre")
            if not entity:
                await update.message.reply_text("Please include a JSON code block with alert definition.")
                return
            code_block = text[entity.offset : entity.offset + entity.length]
            alert_info = AlertInfo(alert_input=AlertInput.model_validate_json(code_block))

            alert_registry: Registry = context.bot_data[Deps.alert_registry]
            if alert_info.alert_input.fn not in alert_registry.alert_map:
                await update.message.reply_text(f"Alert function '{alert_info.alert_input.fn}' is not registered.")
                return

            try:
                await asyncio.run_in_executor(None, alert_registry.get_alert_fn(alert_info.alert_input.fn), alert_info.alert_input,)
            except Exception:
                await update.message.reply_text("Failed to execute alert function. Check your JSON format. 🆘")
                return

            persistent_data_controller: PersistentDataController = context.bot_data[Deps.persistent_data_controller]
            await persistent_data_controller.add_alert(alert_info)
            await update.message.reply_text(f"Alert '{alert_info.alert_input.name}' added successfully. 🆘")
        except Exception:
            await update.message.reply_text("Failed to add alert. Check your JSON format. 🆘")

    @staticmethod
    async def help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Outputs helpful info
        """
        intro = "*Hello, this is financial alert monitoring bot. You can use following commands:*"
        lines: list[str] = [intro]
        for name, obj in Commands.__dict__.items():
            if isinstance(obj, staticmethod):
                fn = obj.__func__
                doc = (fn.__doc__ or "").strip()
                lines.append(f"`{name}` - {doc}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    @staticmethod
    async def reset_timeouts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Resets the timeouts for an alert(s)."""
        persistent_data_controller: PersistentDataController = context.bot_data[Deps.persistent_data_controller]
        try:
            alert_id = None
            if len(update.message.text.split(" ")) == 2:
                alert_id = update.message.text.split(" ")[1]

            alerts: List[AlertInfo] = (
                [a] if (a := await persistent_data_controller.get_alert(alert_id)) else []
                if alert_id
                else await persistent_data_controller.get_alerts()
            )
            if not alerts:
                await update.message.reply_text("Alert(s) not found. 🆘")
                return

            for alert in alerts:
                alert.last_trigger_timestamp_sec = None
            await persistent_data_controller.update_alerts(alerts)
            await update.message.reply_text("Alert(s) timeouts reset successfully.")
        except Exception:
            await update.message.reply_text("Failed to reset alert timeouts.")

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
        self.chat_filter = chat_filter
        self.application.add_handler(CommandHandler("help", Commands.help, filters=chat_filter))
        self.application.add_handler(CommandHandler("logs", Commands.logs, filters=chat_filter))
        self.application.add_handler(CommandHandler("dump_db", Commands.dump_db, filters=chat_filter))
        self.application.add_handler(CommandHandler("alerts", Commands.alerts, filters=chat_filter))
        self.application.add_handler(CommandHandler("functionality", Commands.functionality, filters=chat_filter))
        self.application.add_handler(CommandHandler("reset_timeouts", Commands.reset_timeouts, filters=chat_filter))
        self.application.add_handler(
            CommandHandler(
                "new_alert",
                Commands.new_alert,
                filters=chat_filter
            ),
        )

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
                Commands.show_alert_desc,
            )
        )
        self.application.add_handler(
            MessageHandler(
                chat_filter & filters.Text(strings=["/new_alert"]),
                Commands.new_alert,
            )
        )

        self.application.bot_data[Deps.alert_registry] = alert_registry
        self.application.bot_data[Deps.persistent_data_controller] = persistent_data_controller

        self.application.add_handler(CallbackQueryHandler(self.chart_callback))

    async def chart_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or query.message is None or query.data is None:
            return
        await query.answer()
        # TODO: Maybe add filter for chat_id later.

        persistent_data_controller: PersistentDataController = context.bot_data[Deps.persistent_data_controller]
        alert_button = await persistent_data_controller.get_alert_button(query.data)
        if alert_button is None:
            log.exception("Invalid chart callback payload: %s", query.data)
            return

        alert_registry: Registry = context.bot_data[Deps.alert_registry]
        chart_fn = alert_registry.get_chart_fn(alert_button.fn)
        if chart_fn is None:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=f"Unknown chart function: {alert_button.fn}",
            )
            return

        media_path = None
        try:
            chart = await asyncio.get_running_loop().run_in_executor(None, chart_fn, alert_button)
            media_path = Path(chart.media_path)
            with open(media_path, "rb") as photo:
                await context.bot.send_photo(chat_id=query.message.chat_id, photo=photo)
                media_path.unlink(missing_ok=True)

        except Exception:
            log.exception("Chart function %s failed", alert_button.fn)
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=f"Failed to generate chart for {alert_button.fn}.",
            )
            return
        finally:
            if media_path is not None:
                media_path.unlink(missing_ok=True)

    async def send(self, alert: AlertOutput) -> None:
        if alert.alert_message is None:
            return
        markup = await inline_keyboard_from_buttons(alert.buttons, self.persistent_data_controller)
        if not self.dry_run:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=alert.alert_message.format(),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=markup,
            )
        log.info("Sent alert %s", alert.alert_message.name)

    async def send_many(self, alerts: list[AlertOutput]) -> None:
        for alert in alerts:
            try:
                await self.send(alert)
            except Exception:
                log.exception("Failed to send alert %s", alert.alert_message.name if alert.alert_message else "?")

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
