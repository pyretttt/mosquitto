import functools
from typing import Optional

class Safedict(dict):

    def __init__(
        self, 
        data: Optional[dict] = None, 
        missing_key_formatter: Optional[callable[[str], str]] = None
    ):
        super().__init__(data or {})
        self.missing_key_formatter = missing_key_formatter or (lambda key: "{" + key + "}")


    def __missing__(self, key: str) -> str:
        return self.missing_key_formatter(key)


def resolve_method(root: object, dotted_path: str, allowed_roots: frozenset[str]) -> callable:
    """Walk a dotted attribute path (e.g. ``equity.price.quote``) via getattr.

    Only paths whose first segment is in :data:`ALLOWED_ROOTS` are accepted,
    preventing traversal into internal or dangerous attributes.
    """
    parts = dotted_path.split(".")
    if not parts or parts[0] not in allowed_roots:
        raise ValueError(
            f"Namespace '{parts[0]}' is not in the allow-list"
        )
    call = functools.reduce(getattr, parts, root)
    if not call or not callable(call):
        raise ValueError(
            f"Method '{dotted_path}' is not callable or not found"
        )
    return call

