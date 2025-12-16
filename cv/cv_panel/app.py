import sys


from PySide6.QtCore import QAbstractListModel
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStackedLayout,
    QToolBar,
)


class MainScreen:
    class MainWindow(QMainWindow):
        def __init__(
            self,
        ):
            super().__init__()
            self.setup_ui()

        def setup_ui(self):
            self.setWindowTitle("List of algorithms")

            self.layout = QStackedLayout()
            self.toolbar = QToolBar("My main toolbar")
            self.addToolBar(self.toolbar)

    class Data(QAbstractListModel):
        def __init__(self,):
            pass


class MainWindow(QMainWindow):
    def __init__(
        self,
    ):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("List of algorithms")

        self.layout = QStackedLayout()
        self.toolbar = QToolBar("My main toolbar")
        self.addToolBar(self.toolbar)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
