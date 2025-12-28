from __future__ import annotations

from typing import List

from nicegui import ui

from nicegui_app.state import (
    AppState,
    Checkbox,
    Field,
    Menu,
    NumberField,
    Option,
    ValueSelector,
)


state = AppState()
ui.add_css(
    """
html,
body,
#nicegui-root {
    margin: 0;
    padding: 0;
}

.no-dropdown-icon .q-btn-dropdown__arrow {
    display: none;
}
"""
)


def on_method_selected(method_id: str) -> None:
    state.select_method(method_id)
    methods_sidebar.refresh()
    options_sidebar.refresh()


def on_menu_action(action_id: str | None) -> None:
    if action_id is None:
        return
    state.handle_menu_action(action_id)
    ui.notify(f"Triggered action: {action_id}", position="top", type="positive")


def render_menu_items(items: List[Menu]) -> None:
    """Recursively render nested menu structures."""
    for item in items:
        if item.is_leaf:
            ui.menu_item(item.name, on_click=lambda _=None, action=item.action_id: on_menu_action(action))
        else:
            with ui.menu_item(item.name, auto_close=True):
                with ui.item_section():
                    ui.icon("keyboard_arrow_right")

                with ui.menu().props('anchor="top end" self="top start"'):
                    render_menu_items(item.submenus)


def render_menu(menu: Menu) -> None:
    if menu.is_leaf:
        ui.button(menu.name, on_click=lambda action=menu.action_id: on_menu_action(action)).props("flat color=gray")
    else:
        with ui.dropdown_button(menu.name, auto_close=False).props("flat").classes("h-32px p-1 no-dropdown-icon"):
            render_menu_items(menu.submenus)


@ui.refreshable
def methods_sidebar() -> None:
    with ui.column().classes(
        "w-[160px] h-full bg-[#1e1e1e] text-gray-200 py-3 px-2 gap-2overflow-y-auto border-r border-[#333333]"
    ):
        ui.label("Methods").classes("text-[12px] font-semibold text-gray-400 tracking-wide px-1")
        for method in state.methods:
            is_selected = state.selected_method_id == method.id
            button = ui.button(on_click=lambda m_id=method.id: on_method_selected(m_id)).props("flat")
            button.classes(
                "w-full justify-start items-start text-left px-2 py-1 gap-1 rounded-md "
                "border border-transparent bg-transparent transition-colors duration-200"
            )
            if is_selected:
                button.classes(add="bg-[#2d2d2d] border-[#4d4d4d] text-gray-100")
            else:
                button.classes(add="text-gray-300 hover:bg-[#292929]")
            with button:
                ui.label(method.name).classes("text-[12px] font-medium leading-tight")
                if method.description:
                    ui.label(method.description).classes("text-[10px] text-gray-400 leading-tight")
        ui.space()


def _checkbox_control(option: Option) -> None:
    assert isinstance(option.info, Checkbox)
    ui.checkbox(
        "Enabled",
        value=option.info.value,
        on_change=lambda e, opt=option: setattr(opt.info, "value", bool(e.value)),
    ).props("dense")


def _value_selector_control(option: Option) -> None:
    assert isinstance(option.info, ValueSelector)
    ui.select(
        options=option.info.values,
        value=option.info.selected_value,
        on_change=lambda e, opt=option: opt.info.set_value(e.value),
    ).classes("w-full").props("dense outlined use-input")


def _number_control(option: Option) -> None:
    assert isinstance(option.info, NumberField)
    start_value = option.info.value
    is_int = isinstance(start_value, int)
    step = 1 if is_int else 0.1

    def update_value(e, opt=option) -> None:
        raw_value = e.value
        if raw_value is None or raw_value == "":
            return
        try:
            new_value = int(float(raw_value)) if is_int else float(raw_value)
        except ValueError:
            return
        opt.info.value = opt.info.clamp(new_value)
        e.sender.value = opt.info.value

    ui.number(
        value=start_value,
        min=option.info.min_value,
        max=option.info.max_value,
        step=step,
        on_change=update_value,
    ).classes("w-full").props("dense outlined")


def _field_control(option: Option) -> None:
    assert isinstance(option.info, Field)
    ui.input(
        value=option.info.value,
        on_change=lambda e, opt=option: setattr(opt.info, "value", e.value),
    ).classes("w-full").props("dense outlined")


def render_option_control(option: Option) -> None:
    match option.type:
        case "checkbox":
            _checkbox_control(option)
        case "value_selector":
            _value_selector_control(option)
        case "number_field":
            _number_control(option)
        case "field":
            _field_control(option)
        case _:
            ui.label("Unsupported option").classes("text-red-600")


def render_option_card(option: Option) -> None:
    with ui.card().classes("w-full gap-2 p-4"):
        ui.label(option.name).classes("font-medium")
        if option.description:
            ui.label(option.description).classes("text-xs text-gray-500")
        render_option_control(option)


@ui.refreshable
def options_sidebar() -> None:
    with ui.column().classes("w-80 min-w-[20rem] h-full bg-gray-50 p-4 gap-4 overflow-y-auto border-l border-gray-200"):
        ui.label("Method options").classes("text-base font-semibold")
        options = state.selected_method_options
        if not options:
            ui.label("No options available").classes("text-sm text-gray-500")
        else:
            for option in options:
                render_option_card(option)
        ui.space()


def build_layout() -> None:
    with ui.column().classes("h-screen w-screen bg-gray-50 p-0"):
        with ui.row().classes("w-full h-32px bg-gray-100 px-4 py-0 items-center gap-2"):
            ui.label("Vision Panel").classes("font-semibold text-sm text-gray-800")
            for menu in state.menu:
                render_menu(menu)
            ui.space()
        with ui.column().classes("w-screen flex-1 bg-gray-50"):
            with ui.row().classes("flex-1 w-full overflow-hidden"):
                methods_sidebar()
                ui.element("div").classes("flex-1 h-full bg-white")
                options_sidebar()

        with ui.row().classes("bg-gray-100 h-24px w-full border-t border-gray-200"):
            pass


def main() -> None:
    build_layout()
    ui.query(".nicegui-content").classes("p-0")
    dark_mode = ui.dark_mode()
    dark_mode.enable()
    ui.run(title="Image Transform Panel (NiceGUI)")


if __name__ in ("__main__", "__mp_main__"):
    main()
