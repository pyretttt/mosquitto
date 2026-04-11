from typing import Any
from datetime import datetime, timedelta

from openbb_core.app.model.obbject import OBBject

from src.alert import AlertInput, AlertOutput, AlertMessage, AlertButton
from src.utils import Safedict
from src.app_config import app_config
from src.charts.historical import equity_price_historical


def as_markdown_table(obboject: OBBject) -> str:
    return (
        "```\n"
        + obboject.to_df().T
            .rename_axis("Field")
            .rename(columns={0: "Value"})
            .to_markdown()
        + "\n```"
    )


def output_for_last_result(
    out: OBBject,
    input: AlertInput,
    buttons: bool = True
) -> AlertOutput:
    last_result = out.results[-1]

    if (expression := input.expression) and not expression.evaluate(last_result.model_dump()):
        return AlertOutput(alert_message=None)

    payload = None
    if custom_template := input.custom_template:
        payload = custom_template.format_map(Safedict(last_result.model_dump()))
    else:
        payload = as_markdown_table(out)
    return AlertOutput(
        alert_message=AlertMessage(
            name=input.name,
            expression=input.expression,
            payload=payload,
        ),
        buttons=make_default_buttons(input) if buttons else [],
    )


def make_default_buttons(input: AlertInput) -> list[AlertButton]:
    assert "symbol" in input.params
    common_params = {
        "symbol": input.params["symbol"],
        "timezone": app_config.timezone,
        "extended_hours": True,
    }

    tomorrow = (datetime.now(app_config.zone_info).date() + timedelta(days=1)).isoformat()
    return [
        AlertButton(
            name="Day Chart",
            fn=equity_price_historical.__name__,
            params=dict[str, Any](
                **common_params,
                start_date=datetime.now(app_config.zone_info).date().isoformat(),
                end_date=tomorrow,
                interval="5m",
            ),
        ),
        AlertButton(
            name="Week Chart",
            fn=equity_price_historical.__name__,
            params=dict[str, Any](
                **common_params,
                start_date=(datetime.now(app_config.zone_info).date() - timedelta(days=7)).isoformat(),
                end_date=tomorrow,
                interval="60m",
            ),
        ),
        AlertButton(
            name="Month Chart",
            fn=equity_price_historical.__name__,
            params=dict[str, Any](
                **common_params,
                start_date=(datetime.now(app_config.zone_info).date() - timedelta(days=30)).isoformat(),
                end_date=tomorrow,
                interval="60m",
            ),
        ),
        AlertButton(
            name="Chart Now",
            fn=equity_price_historical.__name__,
            params=dict[str, Any](
                **common_params,
                # start_date=datetime.now(app_config.zone_info).date().isoformat(),
                interval="1d",
            ),
        ),
    ]
