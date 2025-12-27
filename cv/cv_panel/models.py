from typing import List, Optional, Union, Self
from dataclasses import dataclass, field
from numbers import Number
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
    action: MenuActionVariant

    @property
    def is_leaf(self) -> bool:
        match self.action:
            case str():
                return True
            case _:
                return False


def make_default_menu() -> List[Menu]:
    return [
        Menu(name="File", action="save"),
        Menu(
            name="Transforms",
            action=[
                Menu(name="Flip horizontally", action="flip_h"),
                Menu(name="Flip vertically", action="flip_v"),
                Menu(
                    name="Inner level",
                    action=[
                        Menu(name="Flip horizontally", action="flip_h2"),
                        Menu(name="Flip vertically", action="flip_v2"),
                    ],
                ),
            ],
        ),
    ]


@dataclass
class AppState:
    methods: List[Method]
    selected_id: Optional[str] = None
    menu: List[Menu] = field(default_factory=make_default_menu)

    @property
    def selected_method(self) -> Optional[Method]:
        return next((method for method in self.methods if method.id == self.selected_id))

    def option_for_id(self, id: str) -> Optional[Option]:
        return next((option for option in self.selected_method.options if option.id == id))
