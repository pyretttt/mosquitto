from dataclasses import replace

from nicegui_app.state import Screen, Option, CheckboxOption, NumberFieldOption

def run(screen: Screen) -> Screen:
    input = screen.workspace_state.input[0]
    return replace(screen, workspace_state=replace(screen.workspace_state, output=input.convert("L")))


screen = Screen(
    name="Histogram uniformity",
    description="Adjusts histogram histogram for visual improvements",
    options=[
        Option(
            name="Intensity",
            info=NumberFieldOption(value=1.0, min_value=0.0, max_value=5.0),
        ),
        Option(
            name="Verbose",
            info=CheckboxOption(value=True),
        ),
    ],
    run=run,
)