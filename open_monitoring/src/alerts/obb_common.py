from typing import Any
from datetime import datetime, timedelta

from openbb_core.app.model.obbject import OBBject

from src.alert import AlertInput, AlertOutput, AlertMessage, AlertButton
from src.utils import Safedict
from src.app_config import app_config
from src.charts.historical import historical_chart


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
    last_result = out.results[-1].model_dump()

    payload = None
    if custom_template := input.custom_template:
        payload = custom_template.format_map(Safedict(last_result))
    else:
        payload = as_markdown_table(out)

    if input.expression is None:
        return AlertOutput(
            alert_id=input.id,
            alert_message=AlertMessage(
                name=input.name,
                expression=None,
                payload=payload,
            ),
            buttons=make_default_buttons(input) if buttons else [],
        )

    input.expression.data = last_result
    if (expression := input.expression) and not expression.evaluate(last_result):
        return AlertOutput(alert_id=input.id, alert_message=None)

    return AlertOutput(
        alert_id=input.id,
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
        "tickers": input.params["symbol"],
        "prepost": True,
        "keepna": False,
    }

    tomorrow = (datetime.now(app_config.zone_info).date() + timedelta(days=1)).isoformat()
    return [
        AlertButton(
            name="Day Chart",
            fn=historical_chart.__name__,
            params=dict[str, Any](
                **common_params,
                start=datetime.now(app_config.zone_info).date().isoformat(),
                end=tomorrow,
                interval="15m",
            ),
        ),
        AlertButton(
            name="Week Chart",
            fn=historical_chart.__name__,
            params=dict[str, Any](
                **common_params,
                start=(datetime.now(app_config.zone_info).date() - timedelta(days=7)).isoformat(),
                end=tomorrow,
                interval="60m",
            ),
        ),
        AlertButton(
            name="Month Chart",
            fn=historical_chart.__name__,
            params=dict[str, Any](
                **common_params,
                start=(datetime.now(app_config.zone_info).date() - timedelta(days=30)).isoformat(),
                end=tomorrow,
                interval="90m",
            ),
        ),
        AlertButton(
            name="Chart Now",
            fn=historical_chart.__name__,
            params=dict[str, Any](
                **common_params,
                period="1d",
                interval="15m",
            ),
        ),
    ]
