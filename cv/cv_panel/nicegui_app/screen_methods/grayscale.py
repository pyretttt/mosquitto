from dataclasses import replace

from nicegui_app.state import Screen


def run(screen: Screen) -> Screen:
    input = screen.workspace_state.input[0]
    return replace(screen, workspace_state=replace(screen.workspace_state, output=input.convert("L")))
