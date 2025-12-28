from __future__ import annotations
from dataclasses import dataclass, field
import uuid
from numbers import Number
from typing import Optional, List, Union, Self

import reflex as rx


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

    @property
    def current_value(self) -> Number:
        return self._value

    @current_value.setter
    def current_value(self, new_value: Number):
        if new_value > self.max_value:
            raise ValueError("max value exceeded")
        if new_value < self.min_value:
            raise ValueError("min value exceeded")
        self._value = new_value


@dataclass
class ValueSelector:
    values: List[str]
    selected_idx: int = 0

    @property
    def selected_value(self) -> str:
        return self.values[self.selected_idx]

    def set_value(self, new_value: str):
        new_idx = next((i for i, value in enumerate(self.values) if value == new_value))
        if new_idx is None:
            raise ValueError("Wrong value selected")
        self.selected_idx = new_idx


@dataclass
class Field:
    value: str


OptionVariant = Union[NumberField | ValueSelector | Field | Checkbox]


@dataclass
class Option:
    name: str
    info: OptionVariant
    description: Optional[str] = None
    id: str = field(default_factory=make_uuid)


@dataclass
class Method:
    name: str
    description: str
    options: List[Option]
    id: str = field(default_factory=make_uuid)


MenuActionVariant = Union[str | List[Self]]


@dataclass
class Menu:
    name: str
    submenus: List[Menu]
    action_id: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        print("is_leaf: ", self.action_id is not None)
        return self.action_id is not None


def make_default_menu() -> List[Menu]:
    return [
        Menu(name="File", action_id="save"),
        Menu(
            name="Transforms",
            submenus=[
                Menu(name="Flip horizontally", action_id="flip_h"),
                Menu(name="Flip vertically", action_id="flip_v"),
                Menu(
                    name="Inner level",
                    submenus=[
                        Menu(name="Flip horizontally2", action_id="flip_h2"),
                        Menu(name="Flip vertically2", action_id="flip_v2"),
                    ],
                ),
            ],
        ),
    ]


def make_methods():
    return [
        Method(
            name="Image registration",
            description="Using homography, and ransac, to align one image into another",
            options=[
                Option(name="Ransac iterations", info=NumberField(value=20, min_value=1, max_value=1000000)),
                Option(name="Some values", info=ValueSelector(values=["one", "two", "free"])),
            ],
        ),
        Method(name="HUI", description="Pizda", options=[]),
    ]


class AppState(rx.State):
    selected_method_id: Optional[str] = None
    navbar_collapsed: bool = False
    methods: List[Method] = rx.field(default_factory=make_methods)
    menu: List[Menu] = rx.field(default_factory=make_default_menu)

    @rx.event
    def select_method(self, id: str):
        self.selected_method_id = id

    @rx.event
    def trigger_menu_action(self, action_id: str):
        self.last_menu_action = f"Triggered action: {action_id}"

    @rx.event
    def toggle_navbar(self):
        self.navbar_collapsed = not self.navbar_collapsed

    @rx.var
    def selected_method(self) -> Optional[Method]:
        method = next(method for method in self.methods if method.id == self.selected_method_id)
        return method

    @rx.var
    def selected_method_options(self) -> Optional[List[Option]]:
        selected_method = self.select_method
        if selected_method is not None:
            return selected_method.options
        return None
