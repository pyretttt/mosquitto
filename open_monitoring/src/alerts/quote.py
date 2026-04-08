from __future__ import annotations

import operator
from typing import Annotated, Literal, Union

from openbb import obb
from pydantic import BaseModel

from expression import Expression, Op

def quote(params: QuoteParams, expression: Expression) -> str | None:
    q = obb.equity.price.quote(symbol=params.symbol, provider=params.provider).results[-1]
    if expression.evaluate(q.model_dump()):
        return f"{params.symbol} alert: price={q.last_price:.2f} ({q.change_percent:+.1f}%)"
    return None
