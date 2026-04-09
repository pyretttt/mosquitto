from textwrap import dedent

from pydantic import BaseModel, Field

from src.expression import Expression


class ApiInput(BaseModel):
    fn: str
    name: str
    params: dict
    expression: Expression | None = None
    custom_template: str | None = None
    throttle_sec: int = 3600
    sinks: list[str] = Field(default_factory=lambda: ["telegram"])


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


class ApiOutput(BaseModel):
    alert_message: AlertMessage | None = None
