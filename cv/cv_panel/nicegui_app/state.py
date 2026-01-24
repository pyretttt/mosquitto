from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from numbers import Number
from typing import List, Optional, Union, Self, Any, Callable
import uuid
from enum import Enum

from PIL.Image import Image as PILImage
from PIL import Image

from nicegui_app.logs import last_used_logger

DEFAULT_IMAGE_URLS = {
    "Lenna": Image.open("nicegui_app/images/lenna.png"),
    "Text": Image.open("nicegui_app/images/text_example.png"),
    "Stop sign": Image.open("nicegui_app/images/stop_sign.png"),
}


def sort_screens():
    with open("nicegui_app/no_index/last_used_screen", "r") as f:
        while s := f.readline():
            date, name = s.split(":")


def make_uuid() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class MenuAction:
    class ID(str, Enum):
        FileSaved = "saved_file"
        EnableAutoRun = "enable_autorun"
        DisableAutoRun = "disable_autorun"
        FlipVertically = "flip_v"
        FlipHorizontally = "flip_h"
        ImageSelected = "image_selected"

    id: ID
    data: Optional[Any] = None


@dataclass(frozen=True)
class WorkspaceAction:
    class ID(str, Enum):
        FlipInputHorizontally = "flip_h"
        FlipInputVertically = "flip_v"
        UploadedInputImage = "uploaded_input_image"
        Reset = "reset"
        Run = "run"
        SaveFile = "save"

    id: ID
    data: Optional[Any] = None


@dataclass(frozen=True)
class AppAction:
    class ID(str, Enum):
        LeftSideBarVisibilityChanged = "left_sidebar_visibility_changed"
        RightSideBarVisibilityChanged = "right_sidebar_visibility_changed"
        OptionChanged = "option_changed"
        DidSelectScreen = "select_screen"

    id: ID
    data: Optional[Any]


Action = Union[MenuAction, WorkspaceAction, AppAction]


@dataclass(frozen=True)
class Spacer:
    pass


@dataclass(frozen=True)
class Label:
    label: str


FooterWidget = Union[Spacer, Label]


@dataclass(frozen=True)
class CheckboxOption:
    value: bool


@dataclass(frozen=True)
class NumberFieldOption:
    value: Number
    min_value: Number
    max_value: Number

    def clamp(self, new_value: Number) -> Number:
        """Ensure new_value stays inside the configured range."""
        if new_value > self.max_value:
            return self.max_value
        if new_value < self.min_value:
            return self.min_value
        return new_value


@dataclass(frozen=True)
class ValueSelectorOption:
    values: List[str]
    selected_idx: int = 0

    @property
    def selected_value(self) -> str:
        return self.values[self.selected_idx]

    def set_value(self, new_value: str) -> None:
        for idx, value in enumerate(self.values):
            if value == new_value:
                self.selected_idx = idx
                return
        raise ValueError(f"{new_value} is not part of the available values")


@dataclass(frozen=True)
class FieldOption:
    value: str


OptionVariant = Union[NumberFieldOption, ValueSelectorOption, FieldOption, CheckboxOption]


@dataclass(frozen=True)
class Option:
    name: str
    info: OptionVariant
    description: Optional[str] = None
    id: str = field(default_factory=make_uuid)

    @property
    def type(self) -> OptionVariant:
        return self.info


@dataclass(frozen=True)
class Menu:
    name: str
    action: Optional[MenuAction] = None
    submenus: List["Menu"] = field(default_factory=list)
    is_active: bool = True
    id: str = field(default_factory=make_uuid)

    @property
    def is_leaf(self) -> bool:
        return self.action is not None

    @property
    def action_id(self) -> Action.ID:
        return self.action.id


def append_to_default_menu(enabled_auto_run: bool = True, menu: List[Menu] = None) -> List[Menu]:
    return [
        Menu(
            name="Settings",
            submenus=[
                Menu("Disable Autorun", action=MenuAction(id=MenuAction.ID.DisableAutoRun))
                if enabled_auto_run
                else Menu("Enable Autorun", action=MenuAction(id=MenuAction.ID.EnableAutoRun)),
            ],
        ),
        Menu(
            name="File",
            submenus=[Menu(name="Save", action=MenuAction(id=MenuAction.ID.FileSaved))],
        ),
    ] + (menu or [])


def get_default_workspace_actions() -> List[WorkspaceState.Widget]:
    return [
        WorkspaceState.Uploader(name="Upload image"),
        Menu(
            name="Default Images",
            submenus=[
                Menu(name=name, action=MenuAction(id=MenuAction.ID.ImageSelected, data=url))
                for name, url in DEFAULT_IMAGE_URLS.items()
            ],
        ),
        Menu(
            name="Transforms",
            submenus=[
                Menu(name="Flip horizontally", action=MenuAction(id=MenuAction.ID.FlipHorizontally)),
                Menu(name="Flip vertically", action=MenuAction(id=MenuAction.ID.FlipVertically)),
            ],
        ),
        Spacer(),
        WorkspaceState.Button(WorkspaceAction(id=WorkspaceAction.ID.Reset), name="Reset", icon="replay"),
        WorkspaceState.Button(WorkspaceAction(id=WorkspaceAction.ID.Run), name="Run", icon="play_arrow"),
        WorkspaceState.Button(WorkspaceAction(id=WorkspaceAction.ID.SaveFile), name="Save", icon="save"),
    ]


@dataclass(frozen=True)
class WorkspaceState:
    class Layout(Enum):
        OneDimensional = 1

    Spacer = Spacer
    Menu = Menu

    @dataclass(frozen=True)
    class Button:
        action: WorkspaceAction
        id: str = field(default_factory=make_uuid)
        name: Optional[str] = None
        icon: Optional[str] = None

    @dataclass(frozen=True)
    class Uploader:
        name: str

    Widget = Union[Uploader, Button, Spacer, Menu]

    widgets: List[Widget] = field(default_factory=get_default_workspace_actions)
    input: List[PILImage] = field(default_factory=list)
    output: List[PILImage] = field(default_factory=list)
    layout: Layout = Layout.OneDimensional


@dataclass(frozen=True)
class Screen:
    name: str
    description: str
    options: List[Option]
    run: Callable[[Self], Self]
    id: str = field(default_factory=make_uuid)
    top_bar_menu: List[Menu] = field(default_factory=append_to_default_menu)
    workspace_state: WorkspaceState = field(default_factory=WorkspaceState)

    def is_layout_allowed(self, layout: WorkspaceState.Layout) -> bool:
        match layout:
            case WorkspaceState.Layout.OneDimensional:
                return True
            case _:
                return False

    def option_with_id(self, id: str) -> Optional[Option]:
        return next((option for option in self.options if option.id == id), None)


def get_screens() -> List[Screen]:
    from nicegui_app.screen_methods import grayscale
    from nicegui_app.screen_methods import connected_components
    from nicegui_app.screen_methods import dft

    return [
        Screen(
            name="Image registration",
            description="Using homography and RANSAC to align one image into another",
            options=[
                Option(
                    name="Ransac iterations",
                    info=NumberFieldOption(value=20, min_value=1, max_value=1_000_000),
                    description="Controls the number of RANSAC trials",
                ),
                Option(
                    name="Optimization strategy",
                    info=ValueSelectorOption(values=["One", "Two", "Free"]),
                    description="Choose between preset strategies",
                ),
                Option(
                    name="Fast mode",
                    info=CheckboxOption(value=False),
                    description="Trades accuracy for speed",
                ),
                Option(
                    name="Fast mode",
                    info=FieldOption(value="123"),
                    description="Trades accuracy for speed",
                ),
            ],
            run=grayscale.run,
        ),
        grayscale.screen,
        connected_components.screen,
        dft.screen,
        Screen(name="Utility", description="Utility screen without options", options=[], run=grayscale.run),
    ]


_screens = get_screens()


@dataclass(frozen=True)
class Footer:
    widgets: List[FooterWidget] = field(default_factory=list)


@dataclass(frozen=True)
class AppState:
    screens: List[Screen] = field(default_factory=lambda: _screens[:])
    selected_screen_id: str = field(default=_screens[0].id)
    last_menu_action: Optional[str] = None
    is_left_sidebar_visible: bool = True
    is_right_sidebar_visible: bool = True
    is_autorun_enabled: bool = True
    footer: Footer = field(default_factory=Footer)

    @property
    def selected_screen(self) -> Optional[Screen]:
        try:
            return next((screen for screen in self.screens if screen.id == self.selected_screen_id))
        except StopIteration:
            return None

    @property
    def selected_screen_options(self) -> List[Option]:
        screen = self.selected_screen
        return screen.options if screen else []

    def run(self, screen: Screen) -> AppState:
        start = time.perf_counter()
        new_screen = screen.run(screen)
        elapsed = time.perf_counter() - start
        new_screens = self.replace_screen(new_screen)

        return replace(
            self,
            screens=new_screens,
            footer=replace(
                self.footer, widgets=[Label(label=f"Last processing time: {elapsed:.6f} seconds"), Spacer()]
            ),
        )

    def replace_screen(self, updated_screen: Screen) -> List[Screen]:
        return [updated_screen if screen.id == updated_screen.id else screen for screen in self.screens]

    def handle(self, action: Action) -> Self:
        match action:
            case AppAction(id=action_id, data=value):
                match action_id:
                    case AppAction.ID.OptionChanged:
                        new_option = value
                        new_options = [
                            new_option if new_option.id == opt.id else opt for opt in self.selected_screen.options
                        ]
                        new_screens = self.replace_screen(replace(self.selected_screen, options=new_options))
                        return replace(self, screens=new_screens)
                    case AppAction.ID.LeftSideBarVisibilityChanged:
                        return replace(self, is_left_sidebar_visible=value)
                    case AppAction.ID.RightSideBarVisibilityChanged:
                        return replace(self, is_right_sidebar_visible="File saved")
                    case AppAction.ID.DidSelectScreen:
                        identifier = value
                        new_screen = replace(self.selected_screen, top_bar_menu=append_to_default_menu())
                        last_used_logger.info(new_screen.name)
                        return replace(self, selected_screen_id=identifier, screens=self.replace_screen(new_screen))
            case MenuAction(id=action_id, data=value):
                match action_id:
                    case MenuAction.ID.FileSaved:
                        return self
                    case MenuAction.ID.FlipVertically:
                        pass
                    case MenuAction.ID.FlipHorizontally:
                        pass
                    case MenuAction.ID.ImageSelected:
                        new_selected_screen = replace(
                            self.selected_screen,
                            workspace_state=replace(self.selected_screen.workspace_state, input=[value]),
                        )
                        new_app_state = replace(self, screens=self.replace_screen(new_selected_screen))
                        new_app_state = (
                            new_app_state.run(new_selected_screen) if self.is_autorun_enabled else new_app_state
                        )
                        return new_app_state
                    case MenuAction.ID.DisableAutoRun:
                        new_screen = replace(
                            self.selected_screen, top_bar_menu=append_to_default_menu(enabled_auto_run=False)
                        )
                        return replace(self, screens=self.replace_screen(new_screen), is_autorun_enabled=False)

                    case MenuAction.ID.EnableAutoRun:
                        new_screen = replace(
                            self.selected_screen, top_bar_menu=append_to_default_menu(enabled_auto_run=True)
                        )
                        return replace(self, screens=self.replace_screen(new_screen), is_autorun_enabled=True)

            case WorkspaceAction(id=action_id, data=value):
                match action_id:
                    case WorkspaceAction.ID.UploadedInputImage:
                        new_selected_screen = replace(
                            self.selected_screen,
                            workspace_state=replace(self.selected_screen.workspace_state, input=[value]),
                        )
                        new_app_state = replace(self, screens=self.replace_screen(new_selected_screen))
                        new_app_state = (
                            new_app_state.run(new_selected_screen) if self.is_autorun_enabled else new_app_state
                        )
                        return new_app_state
                    case WorkspaceAction.ID.Reset:
                        new_selected_screen = replace(
                            self.selected_screen,
                            workspace_state=replace(self.selected_screen.workspace_state, input=[], output=None),
                        )
                        new_screens = self.replace_screen(new_selected_screen)
                        return replace(self, screens=new_screens)
                    case WorkspaceAction.ID.Run:
                        if not len(self.selected_screen.workspace_state.input):
                            return self
                        new_app_state = self.run(self.selected_screen)
                        return new_app_state
                    case WorkspaceAction.ID.SaveFile:
                        return self

        assert False
