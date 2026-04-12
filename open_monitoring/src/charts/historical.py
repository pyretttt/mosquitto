from __future__ import annotations
import tempfile

import yfinance as yf
import mplfinance as mpf

from src.alert import AlertButton, AlertChart
from src.alert_registry import chart_register
from src.app_config import app_config


@chart_register()
def historical_chart(input: AlertButton) -> AlertChart:
    """
    Charts the historical price of a stock.
    """
    assert input.fn == historical_chart.__name__
    assert "tickers" in input.params
    data = yf.download(
        **input.params,
        multi_level_index=False,
        ignore_tz=True,
    )
    data.index = data.index.tz_localize(app_config.timezone)

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as tmp:
        mpf.plot(
            data,
            type='candle',
            style='yahoo',
            volume=True,
            title=f"{input.params['tickers']}",
            savefig=tmp.name,
            figratio=(10, 5),
            tight_layout=True
        )
        return AlertChart(media_path=tmp.name)
