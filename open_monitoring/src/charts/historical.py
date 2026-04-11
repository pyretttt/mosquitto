from __future__ import annotations
import tempfile

import matplotlib.pyplot as plt
from openbb import obb

from src.alert import AlertButton, AlertChart
from src.alert_registry import chart_register


@chart_register()
def equity_price_historical(input: AlertButton) -> AlertChart:
    """
    Charts the historical price of a stock.
    """
    assert input.fn == equity_price_historical.__name__
    out = obb.equity.price.historical(**input.params)
    df = out.to_df()

    symbol_name = input.params.get("symbol")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["close"])
    ax.set_ylabel(f"{symbol_name} price" if symbol_name is not None else "Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, dpi=100)
        tmp.flush()
        plt.close(fig)
        return AlertChart(media_path=tmp.name)
