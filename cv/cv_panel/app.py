import sys
from enum import IntEnum
from dataclasses import dataclass

from PySide6.QtCore import QAbstractListModel
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedLayout, QToolBar, QHBoxLayout


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("List of algorithms")

        self.page_layout = QHBoxLayout()
        self.central_layout = QStackedLayout()
        self.toolbar = QToolBar("My main toolbar")
        self.addToolBar(self.toolbar)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
