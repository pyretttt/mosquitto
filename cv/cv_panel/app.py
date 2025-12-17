import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedLayout, QToolBar, QWidget

from main_screen import MainWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Welcome back")
        self.container_widget = QWidget(self)
        self.stacked_layout = QStackedLayout(self.container_widget)
        self.main_widget = MainWidget(parent=self.container_widget)
        self.stacked_layout.addWidget(self.main_widget)
        self.toolbar = QToolBar("My main toolbar")
        self.addToolBar(self.toolbar)

        self.setCentralWidget(self.container_widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
