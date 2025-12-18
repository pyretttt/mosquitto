from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, QObject
from PySide6.QtGui import QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
    QListView,
    QStyle,
    QHBoxLayout,
)

from signals import CurrentValueProperty


@dataclass(frozen=True)
class CellData:
    id: str
    name: str
    description: str


class DataModel(QAbstractListModel):
    # Custom roles (must be >= Qt.UserRole)
    DataRole = Qt.UserRole + 1

    def __init__(
        self,
        items: Optional[List[CellData]] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._items: list[CellData] = list(items or [])

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
        # elif role == Qt.SizeHintRole:
        #     return QSize(width=-1, height=44)

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def add_items(self, items: List[CellData]):
        self.beginInsertRows(QModelIndex(), len(self._items), len(self._items) + len(items))
        self._items += items
        self.endInsertRows()

    def reset(self, items: List[CellData]):
        self.beginResetModel()
        self._items = items
        self.endResetModel()


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

        name_rect = rect.adjusted(0, 0, 0, -rect.height() // 2)
        description_rect = rect.adjusted(0, rect.height() // 2 - 2, 0, 0)

        f1 = QFont(option.font)
        f1.setBold(True)
        painter.setFont(f1)
        painter.drawText(name_rect, Qt.AlignLeft | Qt.AlignVCenter, model.name)

        f2 = QFont(option.font)
        f2.setBold(False)
        painter.setFont(f2)
        painter.drawText(description_rect, Qt.AlignLeft | Qt.AlignVCenter, model.description)

        painter.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        s.setHeight(56)
        return s


class MainWidget(QWidget):
    def __init__(self, parent: Optional[QWidget], algorithms: CurrentValueProperty[List[CellData]]):
        super().__init__(parent=parent)
        self.algs_view = QListView(parent=self)
        self.algs_view.setUniformItemSizes(False)
        self.algs_model = DataModel(algorithms.value)
        self.algs_view.setModel(self.algs_model)
        self.algs_view.setItemDelegate(CellDelegate(self.algs_view))

        self.options_view = QListView(parent=self)
        self.options_view.setUniformItemSizes(False)
        self.options_model = DataModel()
        self.options_view.setModel(self.options_model)
        self.options_view.setItemDelegate(CellDelegate(self.options_view))

        self.central_view = QWidget(parent=self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.algs_view, 1)
        layout.addWidget(self.central_view, 4)
        layout.addWidget(self.options_view, 1)

        algorithms.connect(self.update_algs)

    def update_algs(self, new_value: List[CellData] = None):
        self.algs_model.reset(new_value)
