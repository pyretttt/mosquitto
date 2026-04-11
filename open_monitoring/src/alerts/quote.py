from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from openbb import obb

from src.alert import AlertInput, AlertOutput, AlertButton
from src.alert_registry import alert_register
from src.charts.historical import equity_price_historical
from src.alerts.obb_common import output_for_last_result
from src.app_config import app_config


@alert_register()
def quote(input: AlertInput) -> AlertOutput:
    """
    Alerts on the last price of a stock.
    """
    assert input.fn == quote.__name__
    assert "symbol" in input.params

    out = obb.equity.price.quote(**input.params)
    alert_output = output_for_last_result(out, input)
    today_date = datetime.now(app_config.zone_info).date().isoformat()
    symbol = input.params["symbol"]
    alert_output.buttons += [
        AlertButton(
            name="Chart (1 Day)",
            fn=equity_price_historical.__name__,
            params=dict[str, Any](
                symbol=symbol,
                timezone=app_config.timezone,
                start_date=(datetime.now(app_config.zone_info).date() - timedelta(days=1)).isoformat(),
                end_date=today_date,
                interval="5m",
                extended_hours=True
            ),
        ),
        AlertButton(
            name="Chart (1 Week)",
            fn=equity_price_historical.__name__,
            params=dict[str, Any](
                symbol=symbol,
                timezone=app_config.timezone,
                start_date=(datetime.now(app_config.zone_info).date() - timedelta(days=7)).isoformat(),
                end_date=today_date,
                interval="60m",
                extended_hours=True
            ),
        ),
        AlertButton(
            name="Chart (1 Month)",
            fn=equity_price_historical.__name__,
            params=dict[str, Any](
                symbol=symbol,
                timezone=app_config.timezone,
                start_date=(datetime.now(app_config.zone_info).date() - timedelta(days=30)).isoformat(),
                end_date=today_date,
                interval="60m",
                extended_hours=True
            ),
        ),
        AlertButton(
            name="Chart Now",
            fn=equity_price_historical.__name__,
            params=dict[str, Any](
                symbol=symbol,
                timezone=app_config.timezone,
                start_date=(datetime.now(app_config.zone_info).date() - timedelta(days=1)).isoformat(),
                end_date=today_date,
                interval="15m",
                extended_hours=True,
            ),
        ),
    ]

    return alert_output
