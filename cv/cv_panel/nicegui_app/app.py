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


class Colors:
    brd = "#444"
    central = "#1F1F1F"
    brand = "#1A1A1A"


state = AppState()
ui.add_css(
    """
.no-dropdown-icon .q-btn-dropdown__arrow {
    display: none;
}
"""
)


def on_method_selected(method_id: str) -> None:
    state.select_method(method_id)
    methods_sidebar.refresh()
    options_sidebar.refresh()


def toggle_left_sidebar():
    state.is_left_sidebar_visible = not state.is_left_sidebar_visible
    build_layout.refresh()


def on_menu_action(action_id: str | None) -> None:
    if action_id is None:
        return
    state.handle_menu_action(action_id)
    ui.notify(f"Triggered action: {action_id}", position="top", type="positive")


def render_menu_items(items: List[Menu]) -> None:
    """Recursively render nested menu structures."""
    for item in items:
        if item.is_leaf:
            ui.menu_item(
                item.name, on_click=lambda _=None, action=item.action_id: on_menu_action(action), auto_close=True
            ).classes("text-gray-200 hover:bg-accent text-xs min-h-[8px]")
        else:
            with ui.menu_item(item.name, auto_close=True).classes("text-gray-200 text-xs min-h-[8px]"):
                with ui.item_section():
                    ui.icon("keyboard_arrow_right").classes("text-gray-500 pl-2")

                with ui.menu().props('anchor="top end" self="top start"'):
                    render_menu_items(item.submenus)


def render_menu(menu: Menu) -> None:
    if menu.is_leaf:
        (
            ui.menu_item(
                menu.name, on_click=lambda _=None, action=menu.action_id: on_menu_action(action), auto_close=True
            )
            .props("flat dense")
            .classes("text-xs tracking-wide text-gray-200 hover:text-white normal-case")
        )
    else:
        with (
            ui.dropdown_button(menu.name, auto_close=False)
            .props("flat dense")
            .classes(
                "h-[32px] px-2 py-0 no-dropdown-icon text-xs tracking-wide text-gray-200 hover:text-white normal-case"
            )
        ):
            render_menu_items(menu.submenus)


@ui.refreshable
def methods_sidebar() -> None:
    with ui.column().classes(f"w-[160px] h-full text-gray-200 px-3 py-1 gap-1 overflow-y-auto") as col:
        col.set_visibility(state.is_left_sidebar_visible)
        ui.dropdown_button("methods").props("flat dense").classes("text-[11px] uppercase text-gray-300 px-1")
        for method in state.methods:
            is_selected = state.selected_method_id == method.id
            button = ui.button(on_click=lambda m_id=method.id: on_method_selected(m_id)).props("flat dense")
            button.classes("w-full px-2 py-1 rounded-xs normal-case transition-colors duration-150").props(
                'align="left"'
            )
            if is_selected:
                button.classes(add=f"bg-central text-gray-200")
            else:
                button.classes(add="text-gray-300 hover:bg-primary")

            with button:
                ui.label(method.name).classes("text-[12px] font-medium leading-tight text-left")
                if method.description:
                    ui.label(method.description).classes("text-[10px] text-gray-400 leading-tight text-left")


def _checkbox_control(option: Option) -> None:
    assert isinstance(option.info, Checkbox)
    (
        ui.checkbox(
            "Enabled",
            value=option.info.value,
            on_change=lambda e, opt=option: setattr(opt.info, "value", bool(e.value)),
        )
        .props("dense")
        .classes("text-gray-200")
    )


def _value_selector_control(option: Option) -> None:
    assert isinstance(option.info, ValueSelector)
    ui.select(
        options=option.info.values,
        value=option.info.selected_value,
        on_change=lambda e, opt=option: opt.info.set_value(e.value),
    ).classes("w-full text-gray-100").props("dense filled dark use-input")


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
    ).classes("w-full text-gray-100").props("dense filled dark")


def _field_control(option: Option) -> None:
    assert isinstance(option.info, Field)
    ui.input(
        value=option.info.value,
        on_change=lambda e, opt=option: setattr(opt.info, "value", e.value),
    ).classes("w-full text-gray-100").props("dense filled dark")


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
    with ui.card().classes("w-full gap-2 p-4 panel-card text-gray-200"):
        ui.label(option.name).classes("font-medium text-gray-100")
        if option.description:
            ui.label(option.description).classes("text-xs text-gray-400")
        render_option_control(option)


@ui.refreshable
def options_sidebar() -> None:
    with ui.column().classes(f"w-[160px] h-full px-4 py-6 gap-4 overflow-y-auto"):
        ui.label("Settings").classes("text-sm font-semibold text-[#c5c5c5] tracking-[0.3em]")
        options = state.selected_method_options
        if not options:
            ui.label("No options available").classes("text-sm text-gray-400")
        else:
            for option in options:
                render_option_card(option)


@ui.refreshable
def build_layout() -> None:
    with ui.column().classes("h-screen w-screen text-gray-200 p-0 gap-0"):
        with ui.row().classes(f"w-full px-4 bg-brand items-center gap-2 text-xs tracking-wide"):
            with (
                ui.button(on_click=toggle_left_sidebar).props("flat dense").classes("bg-transparent h-[32px] w-[32px]")
            ):
                ui.icon("toggle_on" if state.is_left_sidebar_visible else "toggle_off").classes(
                    "text-gray-500 px-2 text-2xl"
                )
            for menu in state.menu:
                render_menu(menu)
            ui.space()
        with ui.row().classes("flex-1 w-full bg-brand overflow-hidden gap-0"):
            methods_sidebar()
            with ui.column().classes(
                "flex-1 h-full bg-central gap-2 p-3 text-gray-400 gap-0 overflow-hidden rounded-md"
            ):
                ui.label("workspace.ts").classes("text-xs text-[#6f6f6f] tracking-[0.3em]")
                ui.element("div").classes(
                    f"flex-1 w-full rounded-lg border border-dashed border-[{Colors.brd}] bg-[#1b1b1f]"
                )
            options_sidebar()

        with ui.row().classes("h-[24px] w-full text-white px-4 items-center gap-3 text-xs bg-brand"):
            ui.label("Footer")


def main() -> None:
    ui.colors(primary="#888", accent="#202020", brand=Colors.brand, central=Colors.central, brd=Colors.brd)
    build_layout()
    ui.query(".nicegui-content").classes("p-0")
    dark_mode = ui.dark_mode()
    dark_mode.enable()
    ui.run(title="Image Transform Panel (NiceGUI)")


if __name__ in ("__main__", "__mp_main__"):
    main()
