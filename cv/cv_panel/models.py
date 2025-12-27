from typing import List, Optional, Union
from dataclasses import dataclass
from numbers import Number


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
    id: str
    name: str
    description: Optional[str]
    info: OptionVariant


@dataclass
class Method:
    id: str
    title: str
    description: str
    options: List[Option]


@dataclass
class AppState:
    methods: List[Method]
    selected_id: Optional[str]

    @property
    def selected_method(self) -> Optional[Method]:
        return next((method for method in self.methods if method.id == self.selected_id))

    def option_for_id(self, id: str) -> Optional[Option]:
        return next((option for option in self.selected_method.options if option.id == id))
