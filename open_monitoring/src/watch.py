from __future__ import annotations

import operator
from typing import Union

from pydantic import BaseModel
from openbb import obb

_OPS = {
    "<":  operator.lt,
    "<=": operator.le,
    ">":  operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


class QuoteParams(BaseModel):
    symbol: str
    provider: str


class QuoteOp(BaseModel):
    key: str
    op: str
    value: float

    def evaluate(self, data: dict) -> bool:
        fn = _OPS.get(self.op)
        if fn is None:
            raise ValueError(f"Unknown operator: {self.op!r}")
        return fn(data[self.key], self.value)


class QuoteExpression(BaseModel):
    logic: str  # "and" | "or"
    operands: list[Union[QuoteOp, QuoteExpression]]

    def evaluate(self, data: dict) -> bool:
        results = (op.evaluate(data) for op in self.operands)
        return all(results) if self.logic == "and" else any(results)


QuoteExpression.model_rebuild()


def quote_price(params: QuoteParams, expression: QuoteExpression) -> str | None:
    q = obb.equity.price.quote(symbol=params.symbol, provider=params.provider).results[-1]
    if expression.evaluate(q.model_dump()):
        return f"{params.symbol} alert: price={q.last_price:.2f} ({q.change_percent:+.1f}%)"
    return None
