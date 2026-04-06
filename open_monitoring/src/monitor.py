from pydantic import BaseModel


class Pull(BaseModel):
    method: str
    params: dict


class TelegramNotify(BaseModel):
    template: str


class Notify(BaseModel):
    telegram: TelegramNotify


class Monitor(BaseModel):
    """
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
                "template": "AAPL alert: price=${data.last_price}"
            }
        }
    }
    """
    name: str
    pull: Pull
    condition: str
    notify: Notify
