from __future__ import annotations

from typing import List
import base64
from dataclasses import replace

from nicegui import ui
from nicegui.events import KeyEventArguments, UploadEventArguments

from nicegui_app.state import (
    AppState,
    CheckboxOption,
    FieldOption,
    Menu,
    NumberFieldOption,
    Option,
    ValueSelectorOption,
    MenuAction,
    WorkspaceState,
    AppAction,
    WorkspaceAction,
    Action,
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
}
.my-uploader .q-uploader__header {
  background: transparent;
  border-bottom: none;
}
.my-uploader .q-uploader__list {
  display: none;
}
.my-uploader .q-uploader__title {
  font-size: 11px;
}
.my-uploader .q-uploader__subtitle {
  font-size: 9px;
}
.nicegui-content {
    padding: 0px;
}
"""
)


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
    global state
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
    global state
    state.images.input_image_src = url
    state.images.selected_default_input = title
    make_image_workspace.refresh()
    ui.notify(f"Selected default: {title}", type="positive")


def default_tooltip(text: str):
    ui.tooltip(text).classes(f"text-xs border border-[{Colors.brd}] bg-accent text-[{Colors.text2}]")


def render_menu_items(items: List[Menu]) -> None:
    """Recursively render nested menu structures."""
    for item in items:
        if item.is_leaf:
            ui.menu_item(
                item.name, on_click=lambda _=None, action=item.action: ui_action_handler(action), auto_close=True
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
                menu.name, on_click=lambda _=None, action=menu.action: ui_action_handler(action), auto_close=True
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
def screens_sidebar() -> None:
    with ui.column().classes(f"w-[160px] h-full px-3 py-1 gap-1 overflow-y-auto") as col:
        col.set_visibility(state.is_left_sidebar_visible)

        ui.dropdown_button("screens").props("flat dense").classes(f"text-[11px] uppercase text-[{Colors.text1}] px-1")
        for screen in state.screens:
            is_selected = state.selected_screen_id == screen.id
            button = ui.button(
                on_click=lambda screen_id=screen.id: ui_action_handler(
                    AppAction(id=AppAction.ID.DidSelectScreen, data=screen_id)
                )
            ).props("flat dense")
            button.classes("w-full px-2 py-1 rounded-xs normal-case transition-colors duration-150").props(
                'align="left"'
            )
            if is_selected:
                button.classes(add=f"bg-effective")
            else:
                button.classes(add="hover:bg-primary")

            with button:
                ui.label(screen.name).classes(f"text-[12px] font-medium text-[{Colors.text2}] leading-tight text-left")
                if screen.description:
                    ui.label(screen.description).classes(f"text-[10px] text-[{Colors.text1}] leading-tight text-left")


def checkbox_control(option: Option) -> None:
    assert isinstance(option.info, CheckboxOption)

    def _change_option_value(option: Option, value: bool) -> None:
        opt = replace(option, info=replace(option.info, value=value))
        ui_action_handler(AppAction(id=AppAction.ID.OptionChanged, data=opt))
        print(state.selected_screen.options)

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
    assert isinstance(option.info, ValueSelectorOption)

    def _change_option_value(option: Option, value: str) -> None:
        global state
        new_idx = next(idx for idx, val in enumerate(option.info.values) if value == val)
        opt = replace(option, info=replace(option.info, selected_idx=new_idx))
        ui_action_handler(AppAction(id=AppAction.ID.OptionChanged, data=opt))
        print(state.selected_screen.options)

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
    assert isinstance(option.info, NumberFieldOption)
    start_value = option.info.value
    is_int = isinstance(start_value, int)
    step = 1 if is_int else 0.1

    def update_value(e, option=option) -> None:
        global state
        raw_value = e.value
        if raw_value is None or raw_value == "":
            return
        try:
            new_value = int(float(raw_value)) if is_int else float(raw_value)
        except ValueError:
            return
        opt = replace(option, info=replace(option.info, value=option.info.clamp(new_value)))
        e.sender.value = opt.info.value
        ui_action_handler(AppAction(id=AppAction.ID.OptionChanged, data=opt))

    ui.number(
        value=start_value,
        min=option.info.min_value,
        max=option.info.max_value,
        step=step,
        on_change=update_value,
    ).classes(f"w-full text-[{Colors.text1}] min-h-[12px] text-[11px]").props("dense filled dark ")


def field_control(option: Option) -> None:
    assert isinstance(option.info, FieldOption)

    def _change_option_value(option: Option, value: str) -> None:
        global state
        opt = replace(option, info=replace(option.info, value=value))
        ui_action_handler(AppAction(id=AppAction.ID.OptionChanged, data=opt))
        print(state.selected_screen.options)

    ui.input(
        value=option.info.value,
        on_change=lambda event, opt=option: _change_option_value(opt, event.value),
    ).classes(f"w-full text-[{Colors.text1}] min-h-[12px] text-[11px]").props("dense filled dark ")


def render_option_control(option: Option) -> None:
    match option.type:
        case CheckboxOption():
            checkbox_control(option)
        case ValueSelectorOption():
            value_selector_control(option)
        case NumberFieldOption():
            number_control(option)
        case FieldOption():
            field_control(option)
        case _:
            ui.label("Unsupported option").classes("text-red-600")


def render_option_card(option: Option) -> None:
    with ui.card().tight().classes(f"w-full gap-1 p-2").props("flat"):
        with ui.row().classes("gap-[4px] flex-1 w-full flex-nowrap"):
            ui.label(option.name).classes(f"text-xs font-medium text-[{Colors.text2}] flex-1")
            with ui.icon("info").classes(f"text-[{Colors.text1}] px-2 text-xl self-start hover:text-[{Colors.text2}]"):
                default_tooltip(option.description or "No info")
        render_option_control(option)


@ui.refreshable
def options_sidebar() -> None:
    with ui.column().classes(f"w-[160px] h-full px-2 py-1 gap-2 overflow-y-auto "):
        ui.label("Settings").classes(f"text-[11px] uppercase font-medium text-[{Colors.text1}] leading-tight text-left")
        if not state.selected_screen_options:
            ui.label("No options available").classes(f"text-sm text-[{Colors.text1}]")
        else:
            for option in state.selected_screen_options:
                render_option_card(option)


def render_workspace_widget(widget: WorkspaceState.Widget) -> None:
    match widget:
        case WorkspaceState.Uploader() as uploader:
            with ui.upload(label=uploader.name, on_upload=handle_upload_input).classes(
                "my-uploader text-xs max-w-[124px]"
            ):
                ui.tooltip("Drop and image").classes(
                    f"text-xs border border-[{Colors.brd}] bg-accent text-[{Colors.text2}]"
                )
        case WorkspaceState.Button() as button:
            ui.button(
                button.name or "",
                icon=button.icon,
                on_click=lambda: ui_action_handler(WorkspaceAction(id=WorkspaceAction.ID.Reset)),
            ).props("flat dense").classes(f"h-[28px] px-1 text-xs text-[{Colors.text1}]")
        case WorkspaceState.Spacer():
            ui.space()
        case WorkspaceState.Menu() as menu:
            render_menu(menu)


@ui.refreshable
def make_image_workspace() -> None:
    match state.selected_screen.workspace_state.layout:
        case WorkspaceState.Layout.OneDimensional:
            with ui.column().classes(f"flex-1 gap-0 h-full bg-effective overflow-hidden rounded-lg"):
                # Toolbar row: upload, defaults, transforms, reset, run
                with ui.row().classes(f"w-full items-center gap-1 p-1"):
                    for widget in state.selected_screen.workspace_state.widgets:
                        render_workspace_widget(widget)

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
                ui.button(
                    on_click=lambda state=state: ui_action_handler(
                        AppAction(id=AppAction.ID.LeftSideBarVisibilityChanged, data=not state.is_left_sidebar_visible)
                    )
                )
                .props("flat dense")
                .classes("bg-transparent h-[32px] w-[32px]")
            ):
                ui.icon("toggle_on" if state.is_left_sidebar_visible else "toggle_off").classes(
                    f"text-[{Colors.text1}] px-2 text-2xl"
                )
            for menu in state.selected_screen.top_bar_menu:
                render_menu(menu)
            ui.space()
        with ui.row().classes("flex-1 w-full bg-brand overflow-hidden gap-0"):
            screens_sidebar()
            make_image_workspace()
            options_sidebar()

        with ui.row().classes("h-[24px] w-full text-white px-4 items-center gap-3 text-xs bg-brand"):
            ui.label("Footer")


def handle_key(e: KeyEventArguments):
    if e.key in ("1", "2", "3", "4") and not e.action.repeat:
        pass


def ui_action_handler(action: Action) -> AppState:
    global state
    state = state.handle(action)

    match action:
        case AppAction(id=action_id, data=_):
            match action_id:
                case AppAction.ID.OptionChanged:
                    options_sidebar.refresh()
                case AppAction.ID.LeftSideBarVisibilityChanged | AppAction.ID.RightSideBarVisibilityChanged:
                    build_layout.refresh()
                case AppAction.ID.DidSelectScreen:
                    options_sidebar.refresh()
                    screens_sidebar.refresh()
        case MenuAction(id=action_id, data=_):
            ui.notify(f"Triggered action: {action_id.value}", position="bottom", type="positive")
            print("trigger action")
        case WorkspaceAction(id=action_id, data=value):
            match action.id:
                case WorkspaceAction.ID.Reset:
                    make_image_workspace.refresh()
                    ui.notify("Workspace reset", type="warning")
                case _:
                    pass


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
    dark_mode = ui.dark_mode()
    dark_mode.enable()
    ui.run(title="Computer Vision Panel")


if __name__ in ("__main__", "__mp_main__"):
    main()
