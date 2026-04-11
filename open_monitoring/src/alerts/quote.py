from __future__ import annotations

from typing import Any

from openbb import obb

from src.alert import AlertInput, AlertOutput, AlertButton
from src.alert_registry import alert_register
from src.charts.historical import historical_price
from src.alerts.obb_common import output_for_last_result


@alert_register()
def quote(input: AlertInput) -> AlertOutput:
    """
    Alerts on the last price of a stock.
    """
    assert input.fn == quote.__name__
    assert "symbol" in input.params

    out = obb.equity.price.quote(**input.params)
    alert_output = output_for_last_result(out, input)
    alert_output.buttons.append(
        AlertButton(
            name="Chart now",
            fn=historical_price.__name__,
            params=dict[str, Any](
                symbol=input.params["symbol"],
            ),
        )
    )
    return alert_output
