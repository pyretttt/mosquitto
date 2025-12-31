from __future__ import annotations

from typing import List
from copy import deepcopy

from nicegui import ui
from nicegui.events import KeyEventArguments, UploadEventArguments
import base64

from nicegui_app.state import (
    AppState,
    Checkbox,
    Field,
    Menu,
    NumberField,
    Option,
    ValueSelector,
    LayoutType,
    OptionChanged,
    DidTapMenuItem,
    DidSelectMethod,
    DidSelectArea,
)


class Colors:
    brd = "#444"
    effective = "#1F1F1F"
    brand = "#1A1A1A"
    caption = "#2A2A2A"
    text1 = "#888"
    text2 = "#BBB"
    accent_background = "#1b1b1f"


state = AppState()
ui.add_css(
    """
.no-dropdown-icon .q-btn-dropdown__arrow {
    display: none;
}
.my-uploader.q-uploader {
  border: none;
  border-radius: 12px;
  box-shadow: none;
  bg-[]
}
.my-uploader .q-uploader__header {
  background: transparent;
  border-bottom: none;
}
.my-uploader .q-uploader__list {
  display: none;
}
"""
)


def on_method_selected(method_id: str) -> None:
    state.handle(DidSelectMethod(identifier=method_id))
    methods_sidebar.refresh()
    options_sidebar.refresh()


def toggle_left_sidebar():
    state.is_left_sidebar_visible = not state.is_left_sidebar_visible
    build_layout.refresh()


def on_menu_action(action_id: str | None) -> None:
    if action_id is None:
        return
    state.handle(DidTapMenuItem(menu_id=action_id))
    ui.notify(f"Triggered action: {action_id}", position="top", type="positive")


def _bytes_to_data_url(data: bytes, filename: str | None = None) -> str:
    """Convert bytes into a data URL; best effort for content type from filename."""
    content_type = "image/png"
    if filename and "." in filename:
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext in ("jpg", "jpeg"):
            content_type = "image/jpeg"
        elif ext in ("png",):
            content_type = "image/png"
        elif ext in ("gif",):
            content_type = "image/gif"
        elif ext in ("webp",):
            content_type = "image/webp"
        elif ext in ("bmp",):
            content_type = "image/bmp"
        elif ext in ("tif", "tiff"):
            content_type = "image/tiff"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{b64}"


def handle_upload_input(e: UploadEventArguments) -> None:
    """Handle uploaded input image and update workspace state."""
    try:
        content = e.content.read() if hasattr(e.content, "read") else e.content  # type: ignore[attr-defined]
        if isinstance(content, bytes):
            state.images.input_image_src = _bytes_to_data_url(content, e.name)
            state.images.selected_default_input = None
            make_image_workspace.refresh()
            ui.notify(f"Loaded {e.name}", type="positive")
        else:
            ui.notify("Unsupported upload content", type="warning")
    except Exception as ex:
        ui.notify(f"Upload failed: {ex}", type="negative")


def set_default_input_image(title: str, url: str) -> None:
    state.images.input_image_src = url
    state.images.selected_default_input = title
    make_image_workspace.refresh()
    ui.notify(f"Selected default: {title}", type="positive")


def reset_workspace() -> None:
    state.images.input_image_src = None
    state.images.output_image_src = None
    state.images.selected_default_input = None
    state.images.selected_default_output = None
    make_image_workspace.refresh()
    ui.notify("Workspace reset", type="warning")


def run_workspace() -> None:
    # Placeholder hook for running current method/transform
    method_name = state.selected_method.name if state.selected_method else "<none>"
    ui.notify(f"Run: {method_name}", type="positive")


def default_tooltip(text: str):
    ui.tooltip(text).classes(f"text-xs border border-[{Colors.brd}] bg-accent text-[{Colors.text2}]")


def render_menu_items(items: List[Menu]) -> None:
    """Recursively render nested menu structures."""
    for item in items:
        if item.is_leaf:
            ui.menu_item(
                item.name, on_click=lambda _=None, action=item.action_id: on_menu_action(action), auto_close=True
            ).classes(f"text-[{Colors.text1}] hover:bg-accent text-xs min-h-[8px]")
        else:
            with ui.menu_item(item.name, auto_close=True).classes(f"text-[{Colors.text1}] text-xs min-h-[8px]"):
                with ui.item_section():
                    ui.icon("keyboard_arrow_right").classes(f"text-[{Colors.text1}] pl-2")

                with ui.menu().props('anchor="top end" self="top start"'):
                    render_menu_items(item.submenus)


def render_menu(menu: Menu) -> None:
    if menu.is_leaf:
        (
            ui.menu_item(
                menu.name, on_click=lambda _=None, action=menu.action_id: on_menu_action(action), auto_close=True
            )
            .props("flat dense")
            .classes(f"text-xs tracking-wide text-[{Colors.text1}] hover:text-white normal-case")
        )
    else:
        with (
            ui.dropdown_button(menu.name, auto_close=False)
            .props("flat dense")
            .classes(
                f"h-[32px] px-2 py-0 no-dropdown-icon text-xs tracking-wide text-[{Colors.text1}] hover:text-white normal-case"
            )
        ):
            render_menu_items(menu.submenus)


@ui.refreshable
def methods_sidebar() -> None:
    with ui.column().classes(f"w-[160px] h-full px-3 py-1 gap-1 overflow-y-auto") as col:
        col.set_visibility(state.is_left_sidebar_visible)
        ui.dropdown_button("methods").props("flat dense").classes(f"text-[11px] uppercase text-[{Colors.text1}] px-1")
        for method in state.methods:
            is_selected = state.selected_method_id == method.id
            button = ui.button(on_click=lambda m_id=method.id: on_method_selected(m_id)).props("flat dense")
            button.classes("w-full px-2 py-1 rounded-xs normal-case transition-colors duration-150").props(
                'align="left"'
            )
            if is_selected:
                button.classes(add=f"bg-effective")
            else:
                button.classes(add="hover:bg-primary")

            with button:
                ui.label(method.name).classes(f"text-[12px] font-medium text-[{Colors.text2}] leading-tight text-left")
                if method.description:
                    ui.label(method.description).classes(f"text-[10px] text-[{Colors.text1}] leading-tight text-left")


def checkbox_control(option: Option) -> None:
    assert isinstance(option.info, Checkbox)

    def _change_option_value(option: Option, value: bool) -> None:
        opt = deepcopy(option)
        opt.info.value = value
        state.handle(OptionChanged(option=opt))
        print(state.selected_method.options)

    return (
        ui.checkbox(
            "Enabled",
            value=option.info.value,
            on_change=lambda event, option=option: _change_option_value(option, event.value),
        )
        .props("dense")
        .classes(f"text-[{Colors.text1}] text-[11px] font-medium")
    )


def value_selector_control(option: Option) -> None:
    assert isinstance(option.info, ValueSelector)

    def _change_option_value(option: Option, value: str) -> None:
        opt = deepcopy(option)
        new_idx = next(idx for idx, val in enumerate(option.info.values) if value == val)
        opt.info.selected_idx = new_idx
        state.handle(OptionChanged(option=opt))
        print(state.selected_method.options)

    (
        ui.select(
            options=option.info.values,
            value=option.info.selected_value,
            on_change=lambda event, option=option: _change_option_value(option, event.value),
        )
        .classes(f"w-full text-[{Colors.text1}] min-h-[12px] text-[11px]")
        .props("dense filled hide-bottom-space options-dense")
    )


def number_control(option: Option) -> None:
    assert isinstance(option.info, NumberField)
    start_value = option.info.value
    is_int = isinstance(start_value, int)
    step = 1 if is_int else 0.1

    def update_value(e, option=option) -> None:
        raw_value = e.value
        if raw_value is None or raw_value == "":
            return
        try:
            new_value = int(float(raw_value)) if is_int else float(raw_value)
        except ValueError:
            return
        opt = deepcopy(option)
        opt.info.value = opt.info.clamp(new_value)
        e.sender.value = opt.info.value
        state.handle(OptionChanged(option=opt))

    ui.number(
        value=start_value,
        min=option.info.min_value,
        max=option.info.max_value,
        step=step,
        on_change=update_value,
    ).classes(f"w-full text-[{Colors.text1}] min-h-[12px] text-[11px]").props("dense filled dark ")


def field_control(option: Option) -> None:
    assert isinstance(option.info, Field)

    def _change_option_value(option: Option, value: str) -> None:
        opt = deepcopy(option)
        opt.info.value = value
        state.handle(OptionChanged(option=opt))
        print(state.selected_method.options)

    ui.input(
        value=option.info.value,
        on_change=lambda event, opt=option: _change_option_value(opt, event.value),
    ).classes(f"w-full text-[{Colors.text1}] min-h-[12px] text-[11px]").props("dense filled dark ")


def render_option_control(option: Option) -> None:
    match option.type:
        case "checkbox":
            checkbox_control(option)
        case "value_selector":
            value_selector_control(option)
        case "number_field":
            number_control(option)
        case "field":
            field_control(option)
        case _:
            ui.label("Unsupported option").classes("text-red-600")


def render_option_card(option: Option) -> None:
    with ui.card().tight().classes(f"w-full gap-1 p-1").props("flat"):
        with ui.row().classes("gap-[4px] flex-1 w-full flex-nowrap"):
            ui.label(option.name).classes(f"text-xs font-medium text-[{Colors.text2}] flex-1")
            with ui.icon("info").classes(f"text-[{Colors.text1}] px-2 text-xl self-start hover:text-[{Colors.text2}]"):
                default_tooltip(option.description or "No info")
        render_option_control(option)


@ui.refreshable
def options_sidebar() -> None:
    with ui.column().classes(f"w-[160px] h-full px-3 py-1 gap-2 overflow-y-auto "):
        ui.label("Settings").classes(f"text-[11px] uppercase font-medium text-[{Colors.text1}] leading-tight text-left")
        if not state.selected_method_options:
            ui.label("No options available").classes(f"text-sm text-[{Colors.text1}]")
        else:
            for option in state.selected_method_options:
                render_option_card(option)


@ui.refreshable
def make_image_workspace() -> None:
    match state.layout_type:
        case LayoutType.OneDimensional:
            with ui.column().classes(f"flex-1 h-full bg-effective overflow-hidden rounded-lg"):
                # Toolbar row: upload, defaults, transforms, reset, run
                with ui.row().classes(f"w-full items-center gap-1 p-1"):
                    # 1) Upload a file
                    ui.upload(label="Upload image", on_upload=handle_upload_input).classes(
                        "my-uploader text-xs max-w-[124px]"
                    )
                    ui.tooltip("Upload an input image").classes(
                        f"text-xs border border-[{Colors.brd}] bg-accent text-[{Colors.text2}]"
                    )

                    # 2) Dropdown with default images
                    with (
                        ui.dropdown_button("Default images", auto_close=True)
                        .props("flat dense")
                        .classes(
                            f"h-[32px] px-2 py-0 no-dropdown-icon text-xs tracking-wide text-[{Colors.text1}] hover:text-white normal-case"
                        )
                    ):
                        ui.menu_item("Default Images").props("disable").classes(f"text-[{Colors.text1}] text-[10px]")
                        for title, url in state.images.default_images.items():
                            ui.menu_item(title, on_click=lambda t=title, u=url: set_default_input_image(t, u)).classes(
                                f"text-[{Colors.text1}] hover:bg-accent text-xs min-h-[8px]"
                            )

                    # 3) Dropdown transforms button (uses existing menu structure)
                    transforms_menu = next((m for m in state.menu if m.name == "Transforms"), None)
                    if transforms_menu is not None:
                        with (
                            ui.dropdown_button("Transforms", auto_close=False)
                            .props("flat dense")
                            .classes(
                                f"h-[32px] px-2 py-0 no-dropdown-icon text-xs tracking-wide text-[{Colors.text1}] hover:text-white normal-case"
                            )
                        ):
                            render_menu_items(transforms_menu.submenus)

                    ui.space()

                    # 4) Plain reset button
                    ui.button("Reset", on_click=reset_workspace).props("flat dense").classes(
                        f"h-[28px] px-2 text-xs text-[{Colors.text1}] hover:text-white"
                    )
                    # 5) Plain run button
                    ui.button("Run", on_click=run_workspace).props("flat dense").classes(
                        f"h-[28px] px-2 text-xs text-[{Colors.text2}] bg-primary/10 hover:bg-primary/20"
                    )

                # Main splitter area
                with ui.splitter(
                    horizontal=False, reverse=False, value=50, limits=(25, 75), on_change=lambda e: ui.notify(e.value)
                ).classes("w-full flex-1 h-full p-1 overflow-hidden bg-effective ") as splitter:
                    with splitter.before:
                        # top pane (e.g., input)
                        with ui.element("div").classes(
                            f"w-full h-full rounded-md border border-dashed border-[{Colors.brd}] flex items-center justify-center text-[{Colors.text1}]"
                        ):
                            ui.label("Input pane")
                    with splitter.after:
                        # bottom pane (e.g., output)
                        with ui.element("div").classes(
                            f"w-full h-full rounded-md border border-dashed border-[{Colors.brd}] flex items-center justify-center text-[{Colors.text1}]"
                        ):
                            ui.label("Output pane")
                    with splitter.separator:
                        with ui.icon("swipe").classes(
                            f"text-[{Colors.text1}] text-2xl hover:text-[{Colors.text2}]"
                        ) as icon:
                            icon.on("dblclick", lambda: setattr(splitter, "value", 50))
                            ui.tooltip("Drag to resize. Double click to reset")

        case _:
            with ui.element("div").classes(
                f"flex-1 w-full rounded-lg border border-dashed border-[{Colors.brd}] bg-[{Colors.accent_background}]"
            ):
                ui.label("Unsupported layout")


@ui.refreshable
def build_layout() -> None:
    with ui.column().classes(f"h-screen w-screen p-0 gap-0"):
        with ui.row().classes(f"w-full px-4 bg-brand items-center gap-2 text-xs tracking-wide"):
            with (
                ui.button(on_click=toggle_left_sidebar).props("flat dense").classes("bg-transparent h-[32px] w-[32px]")
            ):
                ui.icon("toggle_on" if state.is_left_sidebar_visible else "toggle_off").classes(
                    f"text-[{Colors.text1}] px-2 text-2xl"
                )
            for menu in state.menu:
                render_menu(menu)
            ui.space()
        with ui.row().classes("flex-1 w-full bg-brand overflow-hidden gap-0"):
            methods_sidebar()
            # with ui.column().classes(
            #     f"flex-1 h-full bg-effective gap-0 p-3 text-[{Colors.text1}] overflow-hidden rounded-md"
            # ):
            make_image_workspace()
            options_sidebar()

        with ui.row().classes("h-[24px] w-full text-white px-4 items-center gap-3 text-xs bg-brand"):
            ui.label("Footer")


def handle_key(e: KeyEventArguments):
    if e.key in ("1", "2", "3", "4") and not e.action.repeat:
        state.handle(DidSelectArea(index=int(e.key)))


def main() -> None:
    ui.colors(
        primary="#888",
        accent="#202020",
        brand=Colors.brand,
        effective=Colors.effective,
        brd=Colors.brd,
        accent_background=Colors.accent_background,
    )
    ui.keyboard(on_key=handle_key)
    build_layout()
    ui.query(".nicegui-content").classes("p-0")
    dark_mode = ui.dark_mode()
    dark_mode.enable()
    ui.run(title="Image Transform Panel (NiceGUI)")


if __name__ in ("__main__", "__mp_main__"):
    main()
