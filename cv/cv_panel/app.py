import sys
import os
from typing import List
from dataclasses import dataclass
import yaml
import uuid

from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine

from data.sidebar import LeftSidebarCellData
from signals import CurrentValueProperty


@dataclass
class AppData:
    left_side_bar_data: CurrentValueProperty[List[LeftSidebarCellData]]


def main():
    algorithm_cells = []

    algs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "descriptors")
    for root, _, files in os.walk(top=algs_dir):
        for file in files:
            current_path = os.path.join(root, file)
            with open(current_path, "r") as f:
                try:
                    id = uuid.uuid4()
                    config = yaml.safe_load(f)
                    info = config["info"]
                    cell_model = LeftSidebarCellData(id=id, name=info["name"], description=info["description"])
                    algorithm_cells.append(cell_model)
                except:
                    print("Failed to parse config at: ", current_path)

    algorithms = CurrentValueProperty[List[LeftSidebarCellData]](algorithm_cells)
    app_data = AppData(left_side_bar_data=algorithms)

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.load("main.qml")
    engine.rootObjects()[0].setProperty("leftSideBarData", app_data.left_side_bar_data)

    app.exec()


if __name__ == "__main__":
    main()
