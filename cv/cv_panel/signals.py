from typing import TypeVar, Callable, Generic
from dataclasses import dataclass


from PySide6.QtCore import Signal, QObject, Slot, Property
from PySide6.QtQml import QQmlApplicationEngine


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
    signal = Signal(object, arguments=["new_value"])

    def __init__(self, initial_value: T):
        super().__init__()
        self.current_value = CurrentValue(initial_value, signal=self.signal, op=lambda x: x)
        self.signal.connect(self.current_value.new_value)

    @Slot()
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

    def subscribe(self, callable):
        self.signal.connect(slot=callable)


@dataclass
class EngineProperty(CurrentValueProperty, Generic[T]):
    def __init__(self, initial_value: T):
        super().__init__(initial_value)

    def bind(
        self,
        engine: QQmlApplicationEngine,
        key: str,
        resend_current: bool = True,
    ):
        root_objects = engine.rootObjects()
        if not root_objects:
            raise RuntimeError("Failed to load QML: no root objects")

        root = root_objects[0]
        root.setProperty(key, self)
        if resend_current:
            self.send(self.value)


class EnginePropertyV2(QObject, Generic[T]):
    def __init__(self, initial: T):
        super().__init__()
        self.current_value = initial

    value_changed = Signal(object, arguments=["new_value"])

    def value(self) -> T:
        return self.current_value

    @Slot(object)
    def send(self, new_value: T):
        self.current_value = new_value
        self.value_changed.emit(new_value)

    def subscribe(self, subscriber: Callable[[T], None]):
        self.value_changed.connect(subscriber)

    property = Property(object, fget=value, fset=send, notify=value_changed)

    def bindContext(
        self,
        engine: QQmlApplicationEngine,
        key: str,
        resend_current: bool = True,
    ):
        context = engine.rootContext()
        context.setContextProperty(key, self)
        if resend_current:
            self.send(self.value)
