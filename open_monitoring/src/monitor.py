from pydantic import BaseModel


class Pull(BaseModel):
    method: str
    params: dict


class TelegramNotify(BaseModel):
    template: str


class Notify(BaseModel):
    telegram: TelegramNotify


class Monitor(BaseModel):
    name: str
    pull: Pull
    condition: str
    notify: Notify
