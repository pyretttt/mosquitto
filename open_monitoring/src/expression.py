from __future__ import annotations

import operator
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

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


Operand = Annotated[Union[Op, "Expression"], Field(discriminator="type")]


class Expression(BaseModel):
    type: Literal["expr"] = "expr"
    logic: str | None  # "and" | "or"
    operands: list[Operand]

    def evaluate(self, data: dict) -> bool:
        results = (op.evaluate(data) for op in self.operands)
        return all(results) if self.logic == "and" else any(results)


Expression.model_rebuild()
