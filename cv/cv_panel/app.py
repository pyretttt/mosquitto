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
import sys
import uuid
from typing import Any, Dict, List
from threading import Timer

import yaml
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication

from store import Store, AppState


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

    add_item = {
        "type": "SET_CENTER_COLOR",
        "payload": {
            "color": "red",
        },
    }
    t = Timer(1.0, lambda: store.dispatch(add_item))
    t.start()

    app.exec()


if __name__ == "__main__":
    main()
