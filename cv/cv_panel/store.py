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
    right_items: Dict[str, List[Dict[str, Any]]]
    center_color: str

    def selected_left_item_id(self) -> str | None:
        return next((item["id"] for item in self.left_items if item["is_selected"]), None)

    def option(self, id: str) -> dict | None:
        if self.selected_left_item_id() is None:
            return None
        options = self.right_items[self.selected_left_item_id()]
        return next((options for options in options if options["id"] == id), None)


class LeftSidebarModel(QAbstractListModel):
    IdRole = Qt.UserRole + 1
    NameRole = Qt.UserRole + 2
    DescriptionRole = Qt.UserRole + 3
    IsSelectedRole = Qt.UserRole + 4
    DataRole = Qt.UserRole + 5

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
        if role == self.IsSelectedRole:
            return item.get("is_selected", False)
        if role == self.DataRole:
            return item

        return None

    def roleNames(self) -> Dict[int, bytes]:
        return {
            self.IdRole: b"id",
            self.NameRole: b"name",
            self.IsSelectedRole: b"isSelected",
            self.DescriptionRole: b"description",
            self.DataRole: b"data",
        }

    def set_items(self, items: List[Dict[str, Any]]) -> None:
        self.beginResetModel()
        self._items = list(items)
        self.endResetModel()


class RightSidebarModel(QAbstractListModel):
    TypeRole = Qt.UserRole + 1
    NameRole = Qt.UserRole + 2
    CheckedRole = Qt.UserRole + 3
    ValueRole = Qt.UserRole + 4
    DataRole = Qt.UserRole + 5
    ActionRole = Qt.UserRole + 6

    def __init__(self, items: Dict[str, List[Dict[str, Any]]] | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._items: Dict[str, List[Dict[str, Any]]] = dict(items or {})
        self.selected_id = None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if self.selected_id is None:
            return 0
        if parent.isValid():
            return 0
        return len(self._items.get(self.selected_id, []))

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if self.selected_id is None:
            return None
        if not index.isValid():
            return None
        row = index.row()
        options = self._items.get(self.selected_id)
        if row < 0 or row >= len(options):
            return None
        if not options:
            return None

        item = options[row]
        if role == self.TypeRole:
            return item.get("type", "card")
        if role == self.NameRole:
            return item.get("name", "")
        if role == self.CheckedRole:
            return bool(item.get("checked", False))
        if role == self.ValueRole:
            return item.get("value", None)
        if role == self.DataRole:
            return item
        if role == self.ActionRole:
            return {"type": "OPTION_CHANGED", "payload": {"id": item["id"], "value": None}}

        return None

    def roleNames(self) -> Dict[int, bytes]:
        return {
            self.TypeRole: b"type",
            self.NameRole: b"name",
            self.CheckedRole: b"checked",
            self.ValueRole: b"value",
            self.DataRole: b"data",
        }

    def set_items(self, items: Dict[str, List[Dict[str, Any]]]) -> None:
        self.beginResetModel()
        self._items = dict(items)
        self.endResetModel()

    def set_selected_id(self, id: str):
        self.beginResetModel()
        self.selected_id = id
        self.endResetModel()


def reducer(state: AppState, action: Dict[str, Any]) -> AppState:
    t = action.get("type")
    payload = action.get("payload", {})

    if t == "INIT":
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
        new_item = {
            "id": str(uuid.uuid4()),
            "name": payload.get("name", "New item"),
            "description": payload.get("description", "Added from right sidebar"),
        }
        return replace(state, left_items=[new_item] + state.left_items)

    if t == "TOGGLE_RIGHT":
        row = int(payload).get("row", -1)
        if row < 0 or row >= len(state.right_items):
            return state
        new_right = list(state.right_items)
        item = dict(new_right[row])
        item["checked"] = not bool(item.get("checked"))
        new_right[row] = item
        return replace(state, right_items=new_right)

    if t == "SET_CENTER_COLOR":
        color = payload.get("color")
        if not isinstance(color, str) or not color:
            return state
        return replace(state, center_color=color)

    if t == "LEFT_ITEM_TAPPED":
        index = payload.get("index", 0)
        new_left_items = [{**item, "is_selected": False} for item in state.left_items]
        new_left_items[index]["is_selected"] = not state.left_items[index].get("is_selected", False)

        return replace(state, left_items=new_left_items)

    if t == "OPTION_CHANGED":
        option_id = payload.get("id")
        new_option = dict(state.option(id=option_id))
        new_option["value"] = payload.get("value")
        new_options = state.right_items.copy()
        new_options[state.selected_left_item_id()] = [
            new_option if option_id == option.id else option for option in new_options[state.selected_left_item_id()]
        ]
        return replace(state, right_items=new_options)

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
            self._right_model.set_selected_id(new_state.selected_left_item_id())

        if new_state.right_items != self._state.right_items:
            self._right_model.set_items(new_state.right_items)

        if new_state.center_color != self._state.center_color:
            self._center_color = new_state.center_color
            self.centerColorChanged.emit()

        self._state = new_state
