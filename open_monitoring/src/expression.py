from __future__ import annotations

import operator
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

PRECEDENCE = {"or": 1, "and": 2}
SYMBOLS = {"and": " && ", "or": " || "}

OPS = {
    "<":  operator.lt,
    "<=": operator.le,
    ">":  operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


class Op(BaseModel):
    type: Literal["op"] = "op"
    key: str
    op: str
    value: float

    def evaluate(self, data: dict) -> bool:
        fn = OPS.get(self.op)
        if fn is None:
            raise ValueError(f"Unknown operator: {self.op!r}")
        return fn(data[self.key], self.value)

    def format(self, data: dict | None = None) -> str:
        res = f"{self.key}"
        actual_value = data.get(self.key) if data else None
        if actual_value is not None:
            res += f" {actual_value}"
        res += f" {self.op} {self.value}"
        return res

Operand = Annotated[Union[Op, "Expression"], Field(discriminator="type")]


class Expression(BaseModel):
    type: Literal["expr"] = "expr"
    logic: str | None  # "and" | "or"
    operands: list[Operand]
    data: dict | None = None

    def evaluate(self, data: dict) -> bool:
        results = (op.evaluate(data) for op in self.operands)
        return all(results) if self.logic == "and" else any(results)


    def format(self, data: dict | None = None) -> str:
        sep = SYMBOLS[self.logic]
        parts = []
        for op in self.operands:
            s = op.format(data=data or self.data)
            if isinstance(op, Expression) and PRECEDENCE[op.logic] < PRECEDENCE[self.logic]:
                s = f"({s})"
            parts.append(s)
        return sep.join(parts)


Expression.model_rebuild()
