from __future__ import annotations

from dataclasses import dataclass

import celpy


@dataclass
class ConditionEvaluationResult:
    triggered: bool
    expression: str


def evaluate_condition(data: dict, expression: str) -> ConditionEvaluationResult:
    """Evaluate a CEL *expression* with *data* exposed as the ``data`` variable.

    CEL supports dot-access on maps, so ``data.last_price < 150`` works when
    *data* is ``{"last_price": 145.0, ...}``.  Logical operators are ``||``
    and ``&&`` (C-style), which matches the monitor config syntax.
    """
    env = celpy.Environment()
    ast = env.compile(expression)
    prog = env.program(ast)
    activation = {"data": celpy.json_to_cel(data)}
    result = prog.evaluate(activation)

    return ConditionEvaluationResult(
        triggered=bool(result),
        expression=expression
    )
