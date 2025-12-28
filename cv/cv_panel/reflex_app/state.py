from __future__ import annotations

from typing import Any, Optional, TypedDict, Literal, NotRequired

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


class RenderedOption(TypedDict):
    id: str
    name: str
    description: str
    type: Literal["number", "select", "text", "checkbox"]
    value: Any
    min: NotRequired[Any]
    max: NotRequired[Any]
    values: NotRequired[list[str]]


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
    def selected_method_options(self) -> list[RenderedOption]:
        method = get_method(self.selected_method_id)
        if method is None:
            return []
        rendered: list[RenderedOption] = []
        for option in method.options:
            rendered.append(serialize_option(option))
        return rendered


def serialize_option(option: Option) -> RenderedOption:
    info = option.info
    base: RenderedOption = {
        "id": option.id,
        "name": option.name,
        "description": option.description or "",
    }
    match info:
        case NumberField():
            base.update(
                {
                    "type": "number",
                    "value": info.value,
                    "min": info.min_value,
                    "max": info.max_value,
                }
            )
        case ValueSelector():
            base.update(
                {
                    "type": "select",
                    "value": info.selected_value,
                    "values": info.values,
                }
            )
        case Field():
            base.update(
                {
                    "type": "text",
                    "value": info.value,
                }
            )
        case Checkbox():
            base.update(
                {
                    "type": "checkbox",
                    "value": info.value,
                }
            )
        case _:
            base.update(
                {
                    "type": "text",
                    "value": "",
                }
            )
    return base
