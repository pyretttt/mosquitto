from textwrap import dedent

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
    expression: Expression | None = None
    payload: str

    # TODO: add time

    def format(self) -> str:
        expression = ("`" + self.expression.format() + "`") if self.expression else ""
        expression_with_new_line = expression + "\n\n" if expression else ""
        return (
            f"🔔 *{self.name}*"
            + "\n\n"
            + expression_with_new_line
            + self.payload
        )