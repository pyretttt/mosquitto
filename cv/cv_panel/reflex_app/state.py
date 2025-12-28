from __future__ import annotations

from typing import Optional

import reflex as rx

from reflex_app.models import (
    AppState,
    Checkbox,
    Field,
    Method,
    NumberField,
    Option,
    ValueSelector,
    make_default_menu,
)
import reflex_app.methods.image_registration as image_registration


APP_STATE = AppState(
    methods=[
        image_registration.method_state,
        Method(name="HUI", description="Pizda", options=[]),
    ],
    selected_id=image_registration.method_state.id,
    menu=make_default_menu(),
)


def get_method(method_id: Optional[str]) -> Optional[Method]:
    if method_id is None:
        return None
    return next((method for method in APP_STATE.methods if method.id == method_id), None)


class AppViewState(rx.State):
    """Controls top-level UI for the Reflex port."""

    selected_method_id: str = APP_STATE.selected_id or APP_STATE.methods[0].id
    last_menu_action: str = "Ready"
    navbar_collapsed: bool = False

    def select_method(self, method_id: str):
        self.selected_method_id = method_id

    def trigger_menu_action(self, action_id: str):
        self.last_menu_action = f"Triggered action: {action_id}"

    def toggle_navbar(self):
        self.navbar_collapsed = not self.navbar_collapsed

    @rx.var
    def selected_method_name(self) -> str:
        method = get_method(self.selected_method_id)
        return method.name if method else ""

    @rx.var
    def selected_method_description(self) -> str:
        method = get_method(self.selected_method_id)
        return method.description if method else "Select a method to see the details."

    @rx.var
    def selected_method_options(self) -> list[dict[str, str]]:
        method = get_method(self.selected_method_id)
        if method is None:
            return []
        rendered: list[dict[str, str]] = []
        for option in method.options:
            rendered.append(
                {
                    "name": option.name,
                    "value": describe_option(option),
                    "description": option.description or "",
                }
            )
        return rendered


def describe_option(option: Option) -> str:
    info = option.info
    match info:
        case NumberField():
            return f"{info.value} (range {info.min_value}–{info.max_value})"
        case ValueSelector():
            return f"selected: {info.selected_value}"
        case Field():
            return info.value
        case Checkbox():
            return "enabled" if info.value else "disabled"
        case _:
            return ""
