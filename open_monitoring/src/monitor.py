import uuid

from pydantic import BaseModel, Field


class Pull(BaseModel):
    method: str
    params: dict
    root: str


class TelegramNotify(BaseModel):
    data_key: str
    template: str
    min_interval_sec: int


class Notify(BaseModel):
    telegram: TelegramNotify


class Condition(BaseModel):
    data_key: str
    expression: str


class Monitoring(BaseModel):
    """
    Declarative monitor: pull data → evaluate CEL condition → notify.
    Should be represented as:
    {
        "id": "id",
        "name": "APPL stocks",
        "pull": {
            "root": "obb_api",
            "method": "equity.price.quote",
            "params": {
                "provider": "yfinance",
                "symbol": "AAPL"
            }
        },
        "condition": {
            "data_key": "results",
            "expression": "results.last_price < 150 || results.change_percent > 5.0",
        },
        "notify": {
            "telegram": {
                "data_key": "results",
                "min_interval_sec": 60,
                "template": "AAPL alert: price=${price}"
            }
        }
    }
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    pull: Pull
    condition: Condition
    notify: Notify
