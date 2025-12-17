import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedLayout, QToolBar

from main_screen import MainWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.stacked_layout = QStackedLayout(parent=self)
        self.setWindowTitle("List of algorithms")
        self.main_widget = MainWidget(parent=self)
        self.toolbar = QToolBar("My main toolbar")
        self.addToolBar(self.toolbar)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
