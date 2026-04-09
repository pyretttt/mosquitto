from __future__ import annotations

from openbb import obb

from src.alert import AlertConfig, AlertMessage
from src.utils import Safedict
from src.alert_registry import alert_register


@alert_register()
def quote(alert_config: AlertConfig) -> AlertMessage | None:
    assert alert_config.fn == "quote"

    out = obb.equity.price.quote(**alert_config.params)
    last_result = out.results[-1]
    if alert_config.expression.evaluate(last_result.model_dump()):
        payload = None
        if (custom_template := alert_config.custom_template):
            payload = custom_template.format_map(Safedict(last_result.model_dump()))
        else:
            payload = out.to_df().T.to_string(header=False)
        return AlertMessage(
            name=alert_config.name,
            payload=payload
        )
    return None
