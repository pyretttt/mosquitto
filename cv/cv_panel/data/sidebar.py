from dataclasses import dataclass


@dataclass(frozen=True)
class LeftSidebarCellData:
    id: str
    name: str
    description: str
