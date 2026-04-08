from pydantic import BaseModel, Field

from src.expression import Expression


class AlertConfig(BaseModel):
    fn: str
    name: str
    params: dict
    expression: Expression
    custom_template: str | None = None
    throttle_sec: int = 3600
    sinks: list[str] = Field(default_factory=lambda: ["telegram"])


class AlertMessage(BaseModel):
    name: str
    payload: str

    # TODO: add time

    def format(self) -> str:
        return f"""
        🔔 *{self.name}*\n\n
        {self.payload}
        """