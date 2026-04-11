from typing import Any


from __future__ import annotations
import tempfile

import matplotlib.pyplot as plt
from openbb import obb

from src.alert import AlertInput, AlertOutput, AlertButton, AlertChart
from src.alert_registry import alert_register, chart_register
from src.apis.obb_common import output_for_last_result


@chart_register()
def historical_price(input: AlertButton) -> AlertChart:
    """
    Charts the historical price of a stock.
    """
    assert input.fn == historical_price.__name__
    out = obb.equity.price.historical(**input.params)
    df = out.to_df()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["close"])
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, dpi=100)
        tmp.flush()
        plt.close(fig)
        return AlertChart(media_path=tmp.name)


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
            name="Chart",
            fn=historical_price.__name__,
            params=dict[str, Any](
                symbol=input.params["symbol"],
            ),
        )
    )
    return alert_output
