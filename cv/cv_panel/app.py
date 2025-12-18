import sys
import os
from typing import List
from dataclasses import dataclass
import yaml
import uuid

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedLayout, QToolBar, QWidget

from main_screen import MainWidget, CellData
from signals import CurrentValueProperty


@dataclass
class AppData:
    algorithms: CurrentValueProperty[List[CellData]]


class MainWindow(QMainWindow):
    def __init__(self, app_data: AppData):
        super().__init__()
        self.app_data = app_data
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Welcome back")
        self.container_widget = QWidget(self)
        self.stacked_layout = QStackedLayout(self.container_widget)
        self.main_widget = MainWidget(parent=self.container_widget, algorithms=self.app_data.algorithms)
        self.stacked_layout.addWidget(self.main_widget)
        self.toolbar = QToolBar("My main toolbar")
        self.addToolBar(self.toolbar)

        self.setCentralWidget(self.container_widget)


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
                    cell_model = CellData(id=id, name=info["name"], description=info["description"])
                    algorithm_cells.append(cell_model)
                except:
                    print("Failed to parse config at: ", current_path)

    algorithms = CurrentValueProperty[List[CellData]](algorithm_cells)
    app_data = AppData(algorithms=algorithms)

    app = QApplication(sys.argv)
    window = MainWindow(app_data=app_data)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
