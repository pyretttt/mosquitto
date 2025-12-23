import random
import uuid
from dataclasses import dataclass, replace
from typing import Any, Dict, List

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    Property,
    Qt,
    Signal,
    Slot,
)


@dataclass(frozen=True)
class AppState:
    left_items: List[Dict[str, Any]]
    right_items: List[Dict[str, Any]]
    center_color: str


class LeftSidebarModel(QAbstractListModel):
    IdRole = Qt.UserRole + 1
    NameRole = Qt.UserRole + 2
    DescriptionRole = Qt.UserRole + 3

    def __init__(self, items: List[Dict[str, Any]] | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._items: List[Dict[str, Any]] = items or []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row = index.row()
        if row < 0 or row >= len(self._items):
            return None
        item = self._items[row]

        if role == self.IdRole:
            return item.get("id", "")
        if role == self.NameRole:
            return item.get("name", "")
        if role == self.DescriptionRole:
            return item.get("description", "")
        return None

    def roleNames(self) -> Dict[int, bytes]:
        return {self.IdRole: b"id", self.NameRole: b"name", self.DescriptionRole: b"description"}

    def set_items(self, items: List[Dict[str, Any]]) -> None:
        self.beginResetModel()
        self._items = list(items)
        self.endResetModel()


class RightSidebarModel(QAbstractListModel):
    TypeRole = Qt.UserRole + 1
    TextRole = Qt.UserRole + 2
    TitleRole = Qt.UserRole + 3
    BodyRole = Qt.UserRole + 4
    CheckedRole = Qt.UserRole + 5
    ActionTypeRole = Qt.UserRole + 6

    def __init__(self, items: List[Dict[str, Any]] | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._items: List[Dict[str, Any]] = items or []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row = index.row()
        if row < 0 or row >= len(self._items):
            return None
        item = self._items[row]

        if role == self.TypeRole:
            return item.get("type", "card")
        if role == self.TextRole:
            return item.get("text", "")
        if role == self.TitleRole:
            return item.get("title", "")
        if role == self.BodyRole:
            return item.get("body", "")
        if role == self.CheckedRole:
            return bool(item.get("checked", False))
        if role == self.ActionTypeRole:
            return item.get("actionType", "")
        return None

    def roleNames(self) -> Dict[int, bytes]:
        return {
            self.TypeRole: b"type",
            self.TextRole: b"text",
            self.TitleRole: b"title",
            self.BodyRole: b"body",
            self.CheckedRole: b"checked",
            self.ActionTypeRole: b"actionType",
        }

    def set_items(self, items: List[Dict[str, Any]]) -> None:
        self.beginResetModel()
        self._items = list(items)
        self.endResetModel()


def reducer(state: AppState, action: Dict[str, Any]) -> AppState:
    t = action.get("type")

    if t == "INIT":
        payload = action.get("payload", {})
        return replace(
            state,
            left_items=payload.get("left_items", state.left_items),
            right_items=payload.get("right_items", state.right_items),
            center_color=payload.get("center_color", state.center_color),
        )

    if t == "SHUFFLE_LEFT":
        new_left = list(state.left_items)
        random.shuffle(new_left)
        return replace(state, left_items=new_left)

    if t == "ADD_LEFT":
        payload = action.get("payload", {})
        new_item = {
            "id": str(uuid.uuid4()),
            "name": payload.get("name", "New item"),
            "description": payload.get("description", "Added from right sidebar"),
        }
        return replace(state, left_items=[new_item] + state.left_items)

    if t == "TOGGLE_RIGHT":
        row = int(action.get("payload", {}).get("row", -1))
        if row < 0 or row >= len(state.right_items):
            return state
        new_right = list(state.right_items)
        item = dict(new_right[row])
        item["checked"] = not bool(item.get("checked"))
        new_right[row] = item
        return replace(state, right_items=new_right)

    if t == "SET_CENTER_COLOR":
        color = action.get("payload", {}).get("color")
        if not isinstance(color, str) or not color:
            return state
        return replace(state, center_color=color)

    return state


class Store(QObject):
    centerColorChanged = Signal()

    def __init__(self, initial_state: AppState, parent: QObject | None = None):
        super().__init__(parent)
        self._state = initial_state

        self._left_model = LeftSidebarModel(initial_state.left_items, self)
        self._right_model = RightSidebarModel(initial_state.right_items, self)
        self._center_color = initial_state.center_color

    @Property(QObject, constant=True)
    def leftModel(self) -> QObject:
        return self._left_model

    @Property(QObject, constant=True)
    def rightModel(self) -> QObject:
        return self._right_model

    @Property(str, notify=centerColorChanged)
    def centerColor(self) -> str:
        return self._center_color

    @Slot("QVariantMap")
    def dispatch(self, action: Dict[str, Any]) -> None:
        new_state = reducer(self._state, dict(action))

        # Independent updates: only touch what changed.
        if new_state.left_items != self._state.left_items:
            self._left_model.set_items(new_state.left_items)

        if new_state.right_items != self._state.right_items:
            self._right_model.set_items(new_state.right_items)

        if new_state.center_color != self._state.center_color:
            self._center_color = new_state.center_color
            self.centerColorChanged.emit()

        self._state = new_state
