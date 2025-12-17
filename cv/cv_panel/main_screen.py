from enum import IntEnum
from dataclasses import dataclass
from typing import Union, List, Optional

from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, QSize, QObject
from PySide6.QtGui import QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
    QListView,
    QStyle,
    QVBoxLayout,
)


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
        if index.isValid():
            return 0
        return len(self._items)

    def roleNames(self) -> str:
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
            return QSize(200, 44)

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def add_items(self, items: List[CellModel]):
        self.beginInsertRows(QModelIndex(), len(self._items), len(self._items) + len(items))
        self._items += items
        self.endInsertRows()


class CellDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        painter.save()

        model = index.data(DataModel.DataRole)

        # Background/selection handling (let Qt draw the base)
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawPrimitive(QStyle.PE_PanelItemViewItem, opt, painter, opt.widget)

        rect = option.rect.adjusted(10, 6, -10, -6)

        if model.item_type == int(ItemType.SECTION):
            f = QFont(option.font)
            f.setBold(True)
            painter.setFont(f)
            painter.drawText(rect, Qt.AlignVCenter | Qt.AlignLeft, str(model.data.name))
        elif model.item_type == int(ItemType.CELL):
            name_rect = rect.adjusted(0, 0, 0, -rect.height() // 2)
            description_rect = rect.adjusted(0, rect.height() // 2 - 2, 0, 0)

            f1 = QFont(option.font)
            f1.setBold(True)
            painter.setFont(f1)
            painter.drawText(name_rect, Qt.AlignLeft | Qt.AlignVCenter, model.data.name)

            f2 = QFont(option.font)
            f2.setBold(False)
            painter.setFont(f2)
            painter.drawText(description_rect, Qt.AlignLeft | Qt.AlignVCenter, model.data.description)

            painter.restore()


class MainWidget(QWidget):
    def __init__(self, parent: Optional[QWidget]):
        super().__init__(parent=parent)

        self.list_view = QListView(parent=self)
        self.list_view.setUniformItemSizes(False)
        self.model = DataModel(
            [
                CellModel(item_type=ItemType.CELL, data=CellData("Homography", "Projective Homography")),
                CellModel(item_type=ItemType.CELL, data=CellData("Some sheet", "Some sheet")),
            ]
        )
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(CellDelegate(self.list_view))

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_view)
        # self.layout = QVBoxLayout(parent=self)
        # self.layout.addWidget(self.list_view)
