import uuid

from pydantic import BaseModel, Field


class Pull(BaseModel):
    method: str
    params: dict
    root: str


class Notify(BaseModel):
    data_key: str
    template: str
    min_interval_sec: int
    sinks: list[str]


class Condition(BaseModel):
    data_key: str
    expression: str


class Monitoring(BaseModel):
    """
    Declarative monitor: pull data → evaluate CEL condition → notify.
    Should be represented as:
    {
        "id": "id",
        "controller": "obb",
        "name": "APPL stocks",
        "pull": {
            "root": "obb",
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
            "min_interval_sec": 60,
            "data_key": "results",
            "template": "AAPL alert: price=${price}",
            "sinks" : ["telegram"]
        }
    }
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    controller: str
    name: str
    pull: Pull
    condition: Condition
    notify: Notify
