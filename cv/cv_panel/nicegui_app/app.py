from __future__ import annotations

from typing import List, Optional
from dataclasses import replace
from io import BytesIO
from PIL.Image import Image as Image

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
from nicegui_app.utils import bytes_to_data_url


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


def make_image_ui(source: Optional[Image], alt: Optional[str] = None) -> None:
    if not source:
        return ui.label(alt).classes(f"text-sm text-[{Colors.text1}] hover:text-[{Colors.text2}]")
    match source:
        case Image() as img:
            try:
                buf = BytesIO()
                img.save(buf, format="PNG")
                data_url = bytes_to_data_url(buf.getvalue(), filename="image.png")
                ui.image(data_url).props("fit=contain").classes("w-full h-full ")
                return
            except Exception as ex:
                ui.label(f"Image error: {ex}").classes(f"text-sm text-[{Colors.text1}]")
                return
        case _:
            ui.label("Unsupported image type").classes(f"text-sm text-[{Colors.text1}] hover:text-[{Colors.text2}]")


def handle_upload_input(e: UploadEventArguments) -> None:
    """Handle uploaded input image and update workspace state."""
    try:
        content = e.content.read() if hasattr(e.content, "read") else e.content  # type: ignore[attr-defined]
        if isinstance(content, bytes):
            data_url = bytes_to_data_url(content, e.name)
            ui_action_handler(WorkspaceAction(id=WorkspaceAction.ID.SelectedInputImage, data=data_url))
            make_image_workspace_ui.refresh()
            ui.notify(f"Loaded {e.name}", type="positive")
        else:
            ui.notify("Unsupported upload content", type="warning")
    except Exception as ex:
        ui.notify(f"Upload failed: {ex}", type="negative")


def set_default_input_image(title: str, url: str) -> None:
    ui_action_handler(WorkspaceAction(id=WorkspaceAction.ID.SelectedInputImage, data=url))
    make_image_workspace_ui.refresh()
    ui.notify(f"Selected default: {title}", type="positive")


def make_tooltip_ui(text: str):
    ui.tooltip(text).classes(f"text-xs border border-[{Colors.brd}] bg-accent text-[{Colors.text2}]")


def make_menu_items_ui(items: List[Menu]) -> None:
    """Recursively render nested menu structures."""
    for item in items:
        if item.is_leaf:
            ui.menu_item(
                item.name, on_click=lambda _=None, action=item.action: ui_action_handler(action), auto_close=True
            ).classes(f"text-[{Colors.text1}] font-medium hover:bg-accent text-xs min-h-[8px]")
        else:
            with ui.menu_item(item.name, auto_close=True).classes(f"text-[{Colors.text1}] text-xs min-h-[8px]"):
                with ui.item_section():
                    ui.icon("keyboard_arrow_right").classes(f"text-[{Colors.text1}] pl-2")

                with ui.menu().props('anchor="top end" self="top start"'):
                    make_menu_items_ui(item.submenus)


def make_menu_ui(menu: Menu) -> None:
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
            make_menu_items_ui(menu.submenus)


@ui.refreshable
def make_screens_sidebar_ui() -> None:
    with ui.column().classes(f"w-[130px] h-full px-3 py-1 gap-1 overflow-y-auto") as col:
        col.set_visibility(state.is_left_sidebar_visible)

        ui.dropdown_button("screens").props("flat dense").classes(f"text-[11px] uppercase text-[{Colors.text1}] px-1")

        for screen in state.screens:
            button = (
                ui.button(
                    on_click=lambda screen_id=screen.id: ui_action_handler(
                        AppAction(id=AppAction.ID.DidSelectScreen, data=screen_id)
                    )
                )
                .props("flat dense align='left'")
                .classes("w-full px-2 py-1 rounded-xs normal-case transition-colors duration-150")
            )
            if state.selected_screen_id == screen.id:
                button.classes(add=f"bg-effective")
            else:
                button.classes(add="hover:bg-primary")

            with button:
                ui.label(screen.name).classes(f"text-[12px] font-medium text-[{Colors.text2}] leading-tight text-left")
                if screen.description:
                    ui.label(screen.description).classes(f"text-[10px] text-[{Colors.text1}] leading-tight text-left")


def make_checkbox_ui(option: Option) -> None:
    assert isinstance(option.info, CheckboxOption)

    def change_option_value(option: Option, value: bool) -> None:
        opt = replace(option, info=replace(option.info, value=value))
        ui_action_handler(AppAction(id=AppAction.ID.OptionChanged, data=opt))
        print(state.selected_screen.options)

    return (
        ui.checkbox(
            "Enabled",
            value=option.info.value,
            on_change=lambda event, option=option: change_option_value(option, event.value),
        )
        .props("dense")
        .classes(f"text-[{Colors.text1}] text-[11px] font-medium")
    )


def make_value_selector_ui(option: Option) -> None:
    assert isinstance(option.info, ValueSelectorOption)

    def change_option_value(option: Option, value: str) -> None:
        global state
        new_idx = next(idx for idx, val in enumerate(option.info.values) if value == val)
        opt = replace(option, info=replace(option.info, selected_idx=new_idx))
        ui_action_handler(AppAction(id=AppAction.ID.OptionChanged, data=opt))
        print(state.selected_screen.options)

    (
        ui.select(
            options=option.info.values,
            value=option.info.selected_value,
            on_change=lambda event, option=option: change_option_value(option, event.value),
        )
        .classes(f"w-full text-[{Colors.text1}] min-h-[12px] text-[11px]")
        .props("dense filled hide-bottom-space options-dense")
    )


def make_number_field_ui(option: Option) -> None:
    assert isinstance(option.info, NumberFieldOption)
    is_int = isinstance(option.info.value, int)
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
        value=option.info.value,
        min=option.info.min_value,
        max=option.info.max_value,
        step=step,
        on_change=update_value,
    ).classes(f"w-full text-[{Colors.text1}] min-h-[12px] text-[11px]").props("dense filled dark ")


def make_field_control_ui(option: Option) -> None:
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


def make_option_control_ui(option: Option) -> None:
    match option.type:
        case CheckboxOption():
            make_checkbox_ui(option)
        case ValueSelectorOption():
            make_value_selector_ui(option)
        case NumberFieldOption():
            make_number_field_ui(option)
        case FieldOption():
            make_field_control_ui(option)
        case _:
            ui.label("Unsupported option").classes("text-red-600")


def make_render_option_card_ui(option: Option) -> None:
    with ui.card().tight().classes(f"w-full gap-1 p-2").props("flat"):
        with ui.row().classes("gap-[4px] flex-1 w-full flex-nowrap"):
            ui.label(option.name).classes(f"text-xs font-medium text-[{Colors.text2}] flex-1")
            with ui.icon("info").classes(f"text-[{Colors.text1}] px-2 text-xl self-start hover:text-[{Colors.text2}]"):
                make_tooltip_ui(option.description or "No info")
        make_option_control_ui(option)


@ui.refreshable
def make_options_sidebar_ui() -> None:
    with ui.column().classes(f"w-[130px] h-full px-2 py-1 gap-2 overflow-y-auto "):
        ui.label("Settings").classes(f"text-[11px] uppercase font-medium text-[{Colors.text1}] leading-tight text-left")
        if not state.selected_screen_options:
            ui.label("No options available").classes(f"text-sm text-[{Colors.text1}]")
        else:
            for option in state.selected_screen_options:
                make_render_option_card_ui(option)


def make_workspace_widget_ui(widget: WorkspaceState.Widget) -> None:
    match widget:
        case WorkspaceState.Uploader() as uploader:
            with ui.upload(label=uploader.name, on_upload=handle_upload_input).classes(
                "my-uploader text-xs max-w-[124px]"
            ):
                ui.tooltip("Drop and image").classes(
                    f"text-xs border border-[{Colors.brd}] bg-accent text-[{Colors.text2}]"
                )
        case WorkspaceState.Button() as button:
            with (
                ui.button(
                    "" if button.icon is not None else button.name,
                    icon=button.icon,
                    on_click=lambda action=button.action: ui_action_handler(action),
                )
                .props("flat dense")
                .classes(f"h-[28px] px-1 text-xs text-[{Colors.text1}]")
            ):
                make_tooltip_ui(button.name)
        case WorkspaceState.Spacer():
            ui.space()
        case WorkspaceState.Menu() as menu:
            make_menu_ui(menu)


@ui.refreshable
def make_image_workspace_ui() -> None:
    match state.selected_screen.workspace_state.layout:
        case WorkspaceState.Layout.OneDimensional:
            with ui.column().classes(f"flex-1 gap-1 p-1 h-full bg-effective overflow-hidden rounded-lg"):
                with ui.row().classes(f"w-full items-center gap-1"):
                    for widget in state.selected_screen.workspace_state.widgets:
                        make_workspace_widget_ui(widget)

                with ui.element("div").classes(f"flex-1 w-full rounded-md border border-dashed border-[{Colors.brd}] "):
                    if len(state.selected_screen.workspace_state.input):
                        with ui.splitter(
                            horizontal=False,
                            reverse=False,
                            value=50,
                            limits=(15, 85),
                            on_change=lambda e: ui.notify(e.value),
                        ).classes("h-full p-1 overflow-hidden bg-effective ") as splitter:
                            with splitter.before:
                                # top pane (e.g., input)
                                with ui.element("div").classes(
                                    f"w-full h-full flex items-center justify-center text-[{Colors.text1}] pl-2 pr-3"
                                ):
                                    try:
                                        first_input = (
                                            state.selected_screen.workspace_state.input[0]
                                            if state.selected_screen and state.selected_screen.workspace_state.input
                                            else None
                                        )
                                    except Exception:
                                        first_input = None
                                    make_image_ui(first_input, alt="No input image")
                            with splitter.after:
                                # bottom pane (e.g., output)
                                with ui.element("div").classes(
                                    f"w-full h-full rounded-md flex items-center justify-center text-[{Colors.text1}] pl-3 pr-2"
                                ):
                                    output_img = (
                                        state.selected_screen.workspace_state.output if state.selected_screen else None
                                    )
                                    make_image_ui(output_img, alt="Run transform!")
                            with splitter.separator:
                                with ui.icon("swipe").classes(
                                    f"text-[{Colors.text1}] text-md hover:text-[{Colors.text2}] hover:text-xl"
                                ) as icon:
                                    icon.on("dblclick", lambda: setattr(splitter, "value", 50))
                                    make_tooltip_ui("Drag to resize. Double click to reset")
                    else:
                        with ui.element("div").classes("w-full h-full flex items-center justify-center"):
                            ui.label("Select or drop an image").classes(
                                f"text-lg text-[{Colors.text1}] hover:text-[{Colors.text2}] transition-colors duration-150 "
                            )
        case _:
            with ui.element("div").classes(
                f"flex-1 w-full rounded-lg border border-dashed border-[{Colors.brd}] bg-[{Colors.accent_background}]"
            ):
                ui.label("Unsupported layout")


@ui.refreshable
def make_page_ui() -> None:
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
                make_menu_ui(menu)
            ui.space()
        with ui.row().classes("flex-1 w-full bg-brand overflow-hidden gap-0"):
            make_screens_sidebar_ui()
            make_image_workspace_ui()
            make_options_sidebar_ui()

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
                    pass
                case AppAction.ID.LeftSideBarVisibilityChanged | AppAction.ID.RightSideBarVisibilityChanged:
                    make_page_ui.refresh()
                case AppAction.ID.DidSelectScreen:
                    make_page_ui.refresh()
        case MenuAction(id=action_id, data=value):
            match action_id:
                case MenuAction.ID.ImageSelected:
                    print("image selected")
                    make_image_workspace_ui.refresh()
                case MenuAction.ID.DisableAutoRun | MenuAction.ID.EnableAutoRun:
                    make_page_ui.refresh()

            ui.notify(f"Triggered action: {action_id.value}", position="bottom", type="positive")

        case WorkspaceAction(id=action_id, data=value):
            match action.id:
                case WorkspaceAction.ID.Reset:
                    make_image_workspace_ui.refresh()
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
    make_page_ui()
    dark_mode = ui.dark_mode()
    dark_mode.enable()
    ui.run(title="Computer Vision Panel")


if __name__ in ("__main__", "__mp_main__"):
    main()
