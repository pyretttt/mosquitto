import uuid

from pydantic import BaseModel, Field

from src.expression import Expression


class AlertInput(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fn: str
    name: str
    params: dict
    expression: Expression | None = None
    custom_template: str | None = None
    throttle_sec: int = 3600
    sinks: list[str] = Field(default_factory=lambda: ["telegram"])
    is_single_shot: bool = False


class AlertInfo(BaseModel):
    alert_input: AlertInput
    last_trigger_timestamp_sec: int | None = None


class AlertButton(BaseModel):
    name: str
    fn: str
    params: dict = Field(default_factory=dict)


class AlertChart(BaseModel):
    media_path: str


class AlertMessage(BaseModel):
    name: str
    expression: Expression | None = None
    payload: str
    meta: str = ""

    def format(self) -> str:
        expression = ("`" + self.expression.format() + "`") if self.expression else ""
        expression_with_new_line = expression + "\n\n" if expression else ""
        meta_with_prefix_new_line = "\n\n" + self.meta if self.meta else ""
        return (
            f"🔔 *{self.name}*"
            + "\n\n"
            + expression_with_new_line
            + self.payload
            + meta_with_prefix_new_line
        )


class AlertOutput(BaseModel):
    alert_id: str
    alert_message: AlertMessage | None = None
    buttons: list[AlertButton] = Field(default_factory=list)
