from typing import TypeVar, Callable, Generic


from PySide6.QtCore import Signal, QObject


T = TypeVar("T")
V = TypeVar("V")


class CurrentValue(QObject, Generic[T]):
    def __init__(self, initial_value: T, signal: Signal, op):
        super().__init__()
        self._current_value = initial_value
        signal.connect(op(self.new_value))
        self.signal = signal

    @property
    def value(self):
        return self._current_value

    # @Slot(T)
    def new_value(self, value: T):
        self._current_value = value

    def connect(self, callable):
        self.signal.connect(callable)


class CurrentValueProperty(QObject, Generic[T]):
    signal = Signal(T, arguments=["new_value"])

    def __init__(self, initial_value: T):
        super().__init__()
        self.current_value = CurrentValue(initial_value, signal=self.signal, op=lambda x: x)
        self.signal.connect(self.current_value.new_value)

    def send(self, value: T):
        self.signal.emit(value)

    @property
    def value(self) -> T:
        print(self.current_value.value)

        return self.current_value.value

    def as_current_value(self):
        return self.current_value

    def map(self, mapper: Callable[[T], V]) -> CurrentValue[V]:
        def make_mapper(fn):
            def map(value):
                new_value = mapper(value)
                fn(new_value)

            return map

        return CurrentValue[V](initial_value=mapper(self.value), signal=self.signal, op=make_mapper)

    def connect(self, callable):
        self.signal.connect(callable)
