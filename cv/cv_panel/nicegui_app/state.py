from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Number
from typing import List, Optional, Union, Self, Any
import uuid
from enum import Enum

from PIL.Image import Image

DEFAULT_IMAGE_URLS = {
    "Mountains": "https://picsum.photos/id/1015/1200/800",
    "Forest": "https://picsum.photos/id/102/1200/800",
    "City": "https://picsum.photos/id/1011/1200/800",
    "Kitten": "https://placekitten.com/1200/800",
}


def make_uuid() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class MenuAction:
    class ID(str, Enum):
        FileSaved = "saved_file"
        Flip = "flip"

    id: ID
    data: Optional[Any] = None


@dataclass(frozen=True)
class WorkspaceAction:
    class ID(str, Enum):
        FlipInputHorizontally = "flip_h"
        FlipInputVertically = "flip_v"
        SelectedInputImage = "selected_input_image"
        Reset = "reset"
        Run = "run"

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


def append_to_default_menu(menu: List[Menu] = None) -> List[Menu]:
    return [
        Menu(
            name="File",
            submenus=[Menu(name="Save", action=MenuAction(id=MenuAction.ID.FileSaved))],
        ),
        Menu(
            name="Transforms",
            submenus=[
                Menu(name="Flip horizontally", action=MenuAction(id=MenuAction.ID.Flip)),
                Menu(name="Flip vertically", action=MenuAction(id=MenuAction.ID.Flip)),
                Menu(
                    name="Advanced",
                    submenus=[
                        Menu(name="Flip horizontally 2", action=MenuAction(id=MenuAction.ID.Flip)),
                        Menu(name="Flip vertically 2", action=MenuAction(id=MenuAction.ID.Flip)),
                    ],
                ),
            ],
        ),
    ] + (menu or [])


def get_default_workspace_actions() -> List[WorkspaceState.Widget]:
    return [
        WorkspaceState.Uploader(name="Upload image"),
        Menu(
            name="Default Images",
            submenus=[
                Menu(name=name, action=WorkspaceAction(id=WorkspaceAction.ID.SelectedInputImage, data=url))
                for name, url in DEFAULT_IMAGE_URLS.items()
            ],
        ),
        Menu(
            name="Transforms",
            submenus=[
                Menu(name="Flip horizontally", action=WorkspaceAction(id=WorkspaceAction.ID.FlipInputHorizontally)),
                Menu(name="Flip vertically", action=WorkspaceAction(id=WorkspaceAction.ID.FlipInputVertically)),
            ],
        ),
        Spacer(),
        WorkspaceState.Button(WorkspaceAction(id=WorkspaceAction.ID.Reset), icon="replay"),
        WorkspaceState.Button(WorkspaceAction(id=WorkspaceAction.ID.Run), icon="play_arrow"),
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
    input: List[Union[str | Image]] = field(default_factory=list)
    output: Optional[Image] = None
    layout: Layout = Layout.OneDimensional


@dataclass(frozen=True)
class Screen:
    name: str
    description: str
    options: List[Option]
    id: str = field(default_factory=make_uuid)
    top_bar_menu: List[Menu] = field(default_factory=append_to_default_menu)
    workspace_state: WorkspaceState = field(default_factory=WorkspaceState)

    def is_layout_allowed(self, layout: WorkspaceState.Layout) -> bool:
        match layout:
            case WorkspaceState.Layout.OneDimensional:
                return True
            case _:
                return False


def get_screens() -> List[Screen]:
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
        ),
        Screen(
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
        ),
        Screen(
            name="Utility",
            description="Utility screen without options",
            options=[],
        ),
    ]


_screens = get_screens()


@dataclass(frozen=True)
class AppState:
    screens: List[Screen] = field(default_factory=lambda: _screens[:])
    selected_screen_id: str = field(default=_screens[0].id)
    last_menu_action: Optional[str] = None
    is_left_sidebar_visible: bool = True
    is_right_sidebar_visible: bool = True

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

    def handle(self, action: Action) -> Self:
        match action:
            case AppAction(id=action_id, data=value):
                match action_id:
                    case AppAction.ID.OptionChanged:
                        new_option = value
                        new_options = [
                            new_option if new_option.id == opt.id else opt for opt in self.selected_screen.options
                        ]
                        new_screens = [
                            replace(screen, options=new_options) if screen.id == self.selected_screen_id else screen
                            for screen in self.screens
                        ]
                        return replace(self, screens=new_screens)
                    case AppAction.ID.LeftSideBarVisibilityChanged:
                        return replace(self, is_left_sidebar_visible=value)
                    case AppAction.ID.RightSideBarVisibilityChanged:
                        return replace(self, is_right_sidebar_visible="File saved")
                    case AppAction.ID.DidSelectScreen:
                        identifier = value
                        return replace(self, selected_screen_id=identifier)
            case MenuAction(id=action, data=value):
                return self
            case WorkspaceAction(id=action, data=value):
                return self

        # assert False
