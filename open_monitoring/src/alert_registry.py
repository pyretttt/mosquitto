from typing import Callable


class Registry:
    _instance: "Registry | None" = None
    map: dict[str, Callable]

    def __new__(cls) -> "Registry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.map = {}
        return cls._instance

    def register(self, key: str, fn: Callable) -> None:
        self.map[key] = fn

    def get(self, key: str) -> Callable | None:
        return self.map.get(key)

    def __getitem__(self, key: str) -> Callable:
        return self.map[key]

    def __contains__(self, key: str) -> bool:
        return key in self.map


registry = Registry()


def alert_register(prefix: str | None = None, separator: str = "."):
    """Decorator that registers the wrapped function as ``prefix + fn.__name__`` in the global registry."""

    def decorator(fn: Callable) -> Callable:
        full_prefix = prefix + separator if prefix else ""
        registry.register(full_prefix + fn.__name__, fn)
        return fn

    return decorator
