# import sys
# import os
# from typing import List
# from dataclasses import dataclass, asdict
# import yaml
# import uuid

# from PySide6.QtWidgets import QApplication
# from PySide6.QtQml import QQmlApplicationEngine

# from data.sidebar import LeftSidebarCellData
# from signals import EnginePropertyV2


# @dataclass
# class AppData:
#     left_side_bar_data: EnginePropertyV2[List[dict]]


# def main():
#     algorithm_cells = []

#     algs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "descriptors")
#     for root, _, files in os.walk(top=algs_dir):
#         for file in files:
#             current_path = os.path.join(root, file)
#             with open(current_path, "r") as f:
#                 try:
#                     id = str(uuid.uuid4())
#                     config = yaml.safe_load(f)
#                     info = config["info"]
#                     cell_model = LeftSidebarCellData(id=id, name=info["name"], description=info["description"])
#                     algorithm_cells.append(cell_model)
#                 except:
#                     print("Failed to parse config at: ", current_path)

#     serialized_cells = [asdict(cell) for cell in algorithm_cells]
#     app_data = AppData(left_side_bar_data=EnginePropertyV2(initial=serialized_cells))

#     app = QApplication(sys.argv)
#     engine = QQmlApplicationEngine()
#     engine.quit.connect(app.quit)
#     app_data.left_side_bar_data.bindContext(engine, "leftDataSideBar")
#     engine.load("main.qml")

#     app.exec()


# if __name__ == "__main__":
#     main()

from __future__ import annotations

import os
import random
import sys
import uuid
from dataclasses import dataclass, replace
from typing import Any, Dict, List

import yaml
from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    Property,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication


# ----------------------------
# Redux-ish: State + Reducer
# ----------------------------


@dataclass(frozen=True)
class AppState:
    left_items: List[Dict[str, Any]]
    right_items: List[Dict[str, Any]]
    center_color: str


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
            "heightHint": payload.get("heightHint", 92),
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


# ----------------------------
# Qt Models (independent updates)
# ----------------------------


class LeftSidebarModel(QAbstractListModel):
    IdRole = Qt.UserRole + 1
    NameRole = Qt.UserRole + 2
    DescriptionRole = Qt.UserRole + 3
    HeightHintRole = Qt.UserRole + 4

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
        if role == self.HeightHintRole:
            return int(item.get("heightHint", 72))
        return None

    def roleNames(self) -> Dict[int, bytes]:
        return {
            self.IdRole: b"id",
            self.NameRole: b"name",
            self.DescriptionRole: b"description",
            self.HeightHintRole: b"heightHint",
        }

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


# ----------------------------
# Store (dispatch + minimal diff)
# ----------------------------


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


# ----------------------------
# Data loading helpers
# ----------------------------


def _height_hint_from_text(description: str) -> int:
    # Same cell type, different height.
    # Simple heuristic: longer description => taller cell.
    base = 64
    extra = min(180, (len(description) // 60) * 22)
    return base + extra


def load_left_items_from_descriptors(descriptors_dir: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.isdir(descriptors_dir):
        return items

    for root, _, files in os.walk(descriptors_dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                info = config.get("info") or {}
                name = str(info.get("name", os.path.splitext(file)[0]))
                desc = str(info.get("description", ""))
                items.append(
                    {
                        "id": str(uuid.uuid4()),
                        "name": name,
                        "description": desc,
                        "heightHint": _height_hint_from_text(desc),
                    }
                )
            except Exception as e:
                print("Failed to parse config at:", path, "error:", e)

    # Fallback if empty
    if not items:
        for i in range(20):
            desc = "Example item " + ("— more text " * (i % 6))
            items.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": f"Algorithm {i + 1}",
                    "description": desc,
                    "heightHint": _height_hint_from_text(desc),
                }
            )
    return items


def make_default_right_items() -> List[Dict[str, Any]]:
    return [
        {"type": "header", "text": "Right sidebar"},
        {
            "type": "card",
            "title": "Mixed cell types",
            "body": "This list mixes different delegate layouts (card/toggle/button/header) with different heights.",
        },
        {"type": "toggle", "text": "Enable feature X", "checked": True},
        {"type": "toggle", "text": "Enable feature Y", "checked": False},
        {"type": "button", "text": "Shuffle left items", "actionType": "SHUFFLE_LEFT"},
        {"type": "button", "text": "Add left item", "actionType": "ADD_LEFT"},
        {
            "type": "button",
            "text": "Center: dark blue",
            "actionType": "SET_CENTER_COLOR",
            "payload": {"color": "#102030"},
        },
        {
            "type": "button",
            "text": "Center: dark green",
            "actionType": "SET_CENTER_COLOR",
            "payload": {"color": "#102018"},
        },
    ]


# ----------------------------
# App
# ----------------------------


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    descriptors_dir = os.path.join(base_dir, "descriptors")

    initial = AppState(left_items=[], right_items=[], center_color="#2a2a2a")
    store = Store(initial)

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)

    engine.rootContext().setContextProperty("store", store)

    qml_path = os.path.join(base_dir, "main.qml")
    engine.load(qml_path)
    if not engine.rootObjects():
        raise RuntimeError(f"Failed to load QML: {qml_path}")

    # INIT after QML is up (but before user interacts).
    left_items = load_left_items_from_descriptors(descriptors_dir)
    right_items = make_default_right_items()
    store.dispatch(
        {
            "type": "INIT",
            "payload": {
                "left_items": left_items,
                "right_items": right_items,
                "center_color": "#2a2a2a",
            },
        }
    )

    app.exec()


if __name__ == "__main__":
    main()
