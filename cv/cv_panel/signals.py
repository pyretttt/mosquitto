from typing import Generic, TypeVar, Callable


from PySide6.QtCore import (
    Signal,
    QObject,
    Slot
)


T = TypeVar("T")
V = TypeVar("V")

class CurrentValue(QObject):
    def __init__(self, initial_value: T, signal: Signal, op):
        super().__init__()
        self._current_value = initial_value
        signal.connect(op(self.new_value))

    @property
    def value(self):
        self._current_value

    @Slot(T)
    def new_value(self, value: T):
        self._current_value = value


class CurrentValueProperty(QObject):
    signal = Signal(T)

    def __init__(self, initial_value: T):
        super().__init__()
        self.current_value = CurrentValue(initial_value, signal=self.signal)
        self.signal.connect(self.new_value)

    def send(self, value: T):
        self.signal.emit(value)

    @property
    def value(self):
        return self.current_value.value

    def as_current_value(self):
        return self.current_value

    def map(self, mapper: Callable[[T], V]) -> CurrentValue:
        def make_mapper(fn):
            def map(value):
                new_value = mapper(value)
                fn(new_value)
            return map

        return CurrentValue(
            initial_value=mapper(self.value),
            signal=self.signal,
            op=make_mapper
        )