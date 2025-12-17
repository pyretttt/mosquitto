import sys
from enum import IntEnum
from dataclasses import dataclass
from typing import Union, List, Optional

from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, QSize, QObject
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedLayout, QToolBar, QHBoxLayout


class ItemType(IntEnum):
    SECTION = 0
    CELL = 1


@dataclass(frozen=True)
class SectioData:
    name: str


@dataclass(frozen=True)
class CellData:
    name: str
    description: str


@dataclass(frozen=True)
class CellModel:
    item_type: ItemType
    data: Union[SectioData, CellData]


class Data(QAbstractListModel):
    def __init__(
        self,
    ):
        pass


class DataModel(QAbstractListModel):
    # Custom roles (must be >= Qt.UserRole)
    DataRole = Qt.UserRole + 1

    def __init__(
        self,
        items: Optional[List[CellModel]] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._items: list[CellModel] = list(items or [])

    def rowCount(self, index) -> int:
        return len(self._items)

    def roleNames(self):
        return {
            int(self.DataRole): b"data",
        }

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        item = self._items[index.row()]
        if role == self.DataRole:
            return item
        elif role == Qt.SizeHintRole:
            return QSize(200, 88)

    def add_items(self, items: List[CellModel]):
        self.beginInsertRows(QModelIndex(), len(self._items), len(self._items) + len(items))
        self._items += items
        self.endInsertRows()
