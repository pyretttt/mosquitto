from dataclasses import dataclass

from src.monitor import Monitoring

@dataclass
class Notification:
    alert_message: str
    monitor: Monitoring