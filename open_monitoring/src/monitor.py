from uuid import uuid4
from dataclasses import dataclass, Field

from pydantic import BaseModel


class Pull(BaseModel):
    method: str
    params: dict


class TelegramNotify(BaseModel):
    template: str


class Notify(BaseModel):
    telegram: TelegramNotify


class Monitoring(BaseModel):
    """
    Declarative monitor: pull data → evaluate CEL condition → notify.
    Should be represented as:
    {
        "name": "APPL stocks",
        "pull": {
            "lib": "obb",
            "method": "equity.price.quote",
            "params": {
                "provider": "yfinance",
                "symbol": "AAPL"
            }
        },
        "condition": "data.last_price < 150 || data.change_percent > 5.0",
        "notify": {
            "telegram": {
                "template": "AAPL alert: price=${price}"
            }
        }
    }
    """
    id: str = Field[str](default_factory=lambda: str(uuid4()))
    name: str
    pull: Pull
    condition: str
    notify: Notify
