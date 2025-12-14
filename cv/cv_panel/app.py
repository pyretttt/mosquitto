import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QMainWindow,
    QLabel,
    QMenu,
)

class MainWindow(QMainWindow):
    def __init__(self, ):
        super().__init__()
        self.setWindowTitle("List of algorithms")
        self.label = QLabel("Click in this window")
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)
        self.setCentralWidget(self.label)

    def mouseMoveEvent(self, e):
        self.label.setText("mouseMoveEvent")

    def mousePressEvent(self, e):
        self.label.setText("mousePressEvent")

    def mouseReleaseEvent(self, e):
        self.label.setText("mouseReleaseEvent")

    def mouseDoubleClickEvent(self, e):
        self.label.setText("mouseDoubleClickEvent")

    def contextMenuEvent(self, e):
        context = QMenu(self)
        action1 = QAction("test 1", self)
        context.addAction(action1)
        context.addAction(QAction("test 2", self))
        context.addAction(QAction("test 3", self))
        context.exec(e.globalPos())


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()