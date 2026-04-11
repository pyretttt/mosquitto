import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo


@dataclass
class AppConfig:
    chat_id: str = os.environ["CHAT_ID"]
    bot_token: str = os.environ["BOT_TOKEN"]
    dry_run: bool = os.environ["DRY_RUN"] == "true"
    runloop_interval_sec: int = int(os.environ.get("RUNLOOP_INTERVAL_SEC", "60"))
    log_file_path: str = "/runtime/log"
    alerts_db_file_path: str = "/runtime/alerts.db"
    alert_buttons_file_path: str = "/runtime/alert_buttons.db"
    default_alerts_path: str = "configs/default_alerts.json"
    log_file_max_bytes: int = int(os.environ.get("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    timezone: str = os.environ.get("TIMEZONE", "Europe/Moscow")


    @property
    def zone_info(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


app_config = AppConfig()