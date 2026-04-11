from typing import Callable


class Registry:
    _instance: "Registry | None" = None
    alert_map: dict[str, Callable]
    chart_map: dict[str, Callable]

    def __new__(cls) -> "Registry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.alert_map = {}
            cls._instance.chart_map = {}
        return cls._instance

    def get_alert_fn(self, key: str) -> Callable | None:
        return self.alert_map.get(key)

    def get_chart_fn(self, key: str) -> Callable | None:
        return self.chart_map.get(key)


registry = Registry()


def alert_register(prefix: str | None = None, separator: str = "."):
    """Decorator that registers the wrapped function as ``prefix + fn.__name__`` in the global registry."""

    def decorator(fn: Callable) -> Callable:
        full_prefix = prefix + separator if prefix else ""
        registry.alert_map[full_prefix + fn.__name__] = fn
        return fn

    return decorator


def chart_register(prefix: str | None = None, separator: str = "."):
    """Decorator that registers the wrapped function as ``prefix + fn.__name__`` in the global registry."""

    def decorator(fn: Callable) -> Callable:
        full_prefix = prefix + separator if prefix else ""
        registry.chart_map[full_prefix + fn.__name__] = fn
        return fn

    return decorator
