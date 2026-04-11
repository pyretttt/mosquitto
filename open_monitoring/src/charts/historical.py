from __future__ import annotations
import tempfile

import matplotlib.pyplot as plt
import yfinance as yf

from src.alert import AlertButton, AlertChart
from src.alert_registry import chart_register


@chart_register()
def equity_price_historical(input: AlertButton) -> AlertChart:
    """
    Charts the historical price of a stock.
    """
    assert input.fn == equity_price_historical.__name__
    assert "symbol" in input.params

    df = yf.Ticker(input.params["symbol"]).history(
        start=input.params.get("start_date"),
        end=input.params.get("end_date"),
        interval=input.params.get("interval", "5m"),
        prepost=input.params.get("extended_hours", False),
        auto_adjust=True,
    )

    symbol_name = input.params.get("symbol")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Close"])
    ax.set_ylabel(f"{input.params['symbol']} price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, dpi=100)
        tmp.flush()
        plt.close(fig)
        return AlertChart(media_path=tmp.name)
