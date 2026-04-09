from openbb import OBBject

from src.alert import AlertInput, AlertOutput, AlertMessage
from src.openbb_utils import as_markdown_table
from src.utils import Safedict


def output_for_last_result(
    out: OBBject,
    input: AlertInput,
) -> AlertOutput:
    last_result = out.results[-1]

    if (expression := input.expression) and not expression.evaluate(last_result.model_dump()):
        return AlertOutput(alert_message=None)

    payload = None
    if (custom_template := input.custom_template):
        payload = custom_template.format_map(Safedict(last_result.model_dump()))
    else:
        payload = as_markdown_table(out)
    return AlertOutput(
        alert_message=AlertMessage(
            name=input.name,
            expression=input.expression,
            payload=payload,
        )
    )