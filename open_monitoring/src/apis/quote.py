from __future__ import annotations

from openbb import obb

from src.api import ApiInput, ApiOutput
from src.alert_registry import alert_register
from src.apis.obb_common import output_for_last_result


@alert_register()
def quote(input: ApiInput) -> ApiOutput:
    assert input.fn == quote.__name__
    out = obb.equity.price.quote(**input.params)
    return output_for_last_result(out, input)
