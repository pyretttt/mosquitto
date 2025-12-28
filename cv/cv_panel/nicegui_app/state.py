from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Number
from typing import List, Optional, Union
import uuid


def make_uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class Checkbox:
    value: bool


@dataclass
class NumberField:
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


@dataclass
class ValueSelector:
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


@dataclass
class Field:
    value: str


OptionVariant = Union[NumberField, ValueSelector, Field, Checkbox]


@dataclass
class Option:
    name: str
    info: OptionVariant
    description: Optional[str] = None
    id: str = field(default_factory=make_uuid)

    @property
    def type(self) -> str:
        if isinstance(self.info, NumberField):
            return "number_field"
        if isinstance(self.info, ValueSelector):
            return "value_selector"
        if isinstance(self.info, Field):
            return "field"
        if isinstance(self.info, Checkbox):
            return "checkbox"
        raise ValueError("Unsupported option type")


@dataclass
class Method:
    name: str
    description: str
    options: List[Option]
    id: str = field(default_factory=make_uuid)


@dataclass
class Menu:
    name: str
    action_id: Optional[str] = None
    submenus: List["Menu"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.action_id is not None


def make_default_menu() -> List[Menu]:
    return [
        Menu(
            name="File",
            submenus=[
                Menu(name="Save", action_id="save"),
                Menu(name="Save As", action_id="save_as"),
            ],
        ),
        Menu(
            name="Transforms",
            submenus=[
                Menu(name="Flip horizontally", action_id="flip_h"),
                Menu(name="Flip vertically", action_id="flip_v"),
                Menu(
                    name="Advanced",
                    submenus=[
                        Menu(name="Flip horizontally 2", action_id="flip_h2"),
                        Menu(name="Flip vertically 2", action_id="flip_v2"),
                    ],
                ),
            ],
        ),
    ]


def make_methods() -> List[Method]:
    return [
        Method(
            name="Image registration",
            description="Using homography and RANSAC to align one image into another",
            options=[
                Option(
                    name="Ransac iterations",
                    info=NumberField(value=20, min_value=1, max_value=1_000_000),
                    description="Controls the number of RANSAC trials",
                ),
                Option(
                    name="Optimization strategy",
                    info=ValueSelector(values=["One", "Two", "Free"]),
                    description="Choose between preset strategies",
                ),
                Option(
                    name="Fast mode",
                    info=Checkbox(value=False),
                    description="Trades accuracy for speed",
                ),
            ],
        ),
        Method(
            name="Histogram uniformity",
            description="Adjusts histogram histogram for visual improvements",
            options=[
                Option(
                    name="Intensity",
                    info=NumberField(value=1.0, min_value=0.0, max_value=5.0),
                ),
                Option(
                    name="Verbose",
                    info=Checkbox(value=True),
                ),
            ],
        ),
        Method(
            name="HUI",
            description="Utility method without options",
            options=[],
        ),
    ]


class AppState:
    def __init__(self) -> None:
        self.menu: List[Menu] = make_default_menu()
        self.methods: List[Method] = make_methods()
        self.selected_method_id: Optional[str] = self.methods[0].id if self.methods else None
        self.last_menu_action: Optional[str] = None

    def select_method(self, method_id: str) -> None:
        self.selected_method_id = method_id

    def handle_menu_action(self, action_id: str) -> None:
        self.last_menu_action = action_id

    @property
    def selected_method(self) -> Optional[Method]:
        return next((method for method in self.methods if method.id == self.select_method_id))

    @property
    def selected_method_options(self) -> List[Option]:
        method = self.selected_method
        return method.options if method else []
