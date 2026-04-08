from __future__ import annotations

from openbb import obb

from src.alert import AlertConfig, AlertMessage
from src.utils import Safedict


def quote(alert_config: AlertConfig) -> AlertMessage | None:
    out = obb.equity.price.quote(**alert_config.params).results[-1]
    if alert_config.expression.evaluate(out.model_dump()):
        payload = None
        if (custom_template := alert_config.custom_template):
            payload = custom_template.format_map(Safedict(out.model_dump()))
        else:
            payload = out.to_df().T.to_string()
        return AlertMessage(
            name=alert_config.name,
            payload=payload
        )
    return None
