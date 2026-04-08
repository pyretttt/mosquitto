from typing import Callable

class _Registry:
    _instance: "_Registry | None" = None
    _map: dict[str, Callable]

    def __new__(cls) -> "_Registry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._map = {}
        return cls._instance

    def register(self, key: str, fn: Callable) -> None:
        self._map[key] = fn

    def get(self, key: str) -> Callable | None:
        return self._map.get(key)

    def __getitem__(self, key: str) -> Callable:
        return self._map[key]

    def __contains__(self, key: str) -> bool:
        return key in self._map


registry = _Registry()


def alert_register(prefix: str | None = None, separator: str = "."):
    """Decorator that registers the wrapped function as ``prefix + fn.__name__`` in the global registry."""
    def decorator(fn: Callable) -> Callable:
        prefix = prefix + separator if prefix else ""
        registry.register(prefix + fn.__name__, fn)
        return fn
    return decorator