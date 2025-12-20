import sys
import os
from typing import List
from dataclasses import dataclass, asdict
import yaml
import uuid

from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine

from data.sidebar import LeftSidebarCellData
from signals import EnginePropertyV2


@dataclass
class AppData:
    left_side_bar_data: EnginePropertyV2[List[dict]]


def main():
    algorithm_cells = []

    algs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "descriptors")
    for root, _, files in os.walk(top=algs_dir):
        for file in files:
            current_path = os.path.join(root, file)
            with open(current_path, "r") as f:
                try:
                    id = str(uuid.uuid4())
                    config = yaml.safe_load(f)
                    info = config["info"]
                    cell_model = LeftSidebarCellData(id=id, name=info["name"], description=info["description"])
                    algorithm_cells.append(cell_model)
                except:
                    print("Failed to parse config at: ", current_path)

    serialized_cells = [asdict(cell) for cell in algorithm_cells]
    app_data = AppData(left_side_bar_data=EnginePropertyV2(initial=serialized_cells))

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    app_data.left_side_bar_data.bindContext(engine, "leftDataSideBar")
    engine.load("main.qml")

    # Send data to QML using the same structure ListView expects (list of dicts)
    extra_cell = LeftSidebarCellData(id="id", name="info", description="info")
    payload = app_data.left_side_bar_data.get_value() + [asdict(extra_cell)]
    print(payload)
    app_data.left_side_bar_data.send(payload)

    app.exec()


if __name__ == "__main__":
    main()
