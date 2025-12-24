from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Dict, List
import yaml

from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication

from store import Store, AppState


def make_debug_app_state() -> AppState:
    left_items = []
    for i in range(20):
        desc = "Example item " + ("— more text " * (i % 6))
        left_items.append(
            {
                "id": str(uuid.uuid4()),
                "name": f"Algorithm {i + 1}",
                "description": desc,
            }
        )

    right_items = [
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
    return AppState(left_items=left_items, right_items=right_items, center_color="#2a2a2a")


def load_app_state(descriptors_dir: str) -> List[Dict[str, Any]]:
    left_items: List[Dict[str, Any]] = []
    right_items: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(descriptors_dir):
        return make_debug_app_state()

    for root, _, files in os.walk(descriptors_dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                info = config.get("info") or {}
                id = str(uuid.uuid4())
                left_items.append(
                    {
                        "id": id,
                        "name": info.get("name", os.path.splitext(file)[0]),
                        "description": info.get("description", ""),
                    }
                )
                right_items[id] = config.get("options", [])

            except Exception as e:
                print("Failed to parse config at:", path, "error:", e)

    return AppState(
        left_items=left_items,
        right_items=right_items,
        center_color="#2a2a2a",
    )


# ----------------------------
# App
# ----------------------------


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    descriptors_dir = os.path.join(base_dir, "descriptors")

    app_state = load_app_state(descriptors_dir)
    store = Store(app_state)

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.rootContext().setContextProperty("store", store)
    qml_path = os.path.join(base_dir, "main.qml")
    engine.load(qml_path)
    if not engine.rootObjects():
        raise RuntimeError(f"Failed to load QML: {qml_path}")
    # store.dispatch(
    #     {
    #         "type": "INIT",
    #         "payload": {
    #             "left_items": left_items,
    #             "right_items": right_items,
    #             "center_color": "#2a2a2a",
    #         },
    #     }
    # )

    app.exec()


if __name__ == "__main__":
    main()
