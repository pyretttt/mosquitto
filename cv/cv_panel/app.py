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
    QTabWidget,
    QStackedLayout,
    QToolBar,
    QStatusBar
)

class MainWindow(QMainWindow):
    def __init__(self, ):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("List of algorithms")
        
        self.layout = QStackedLayout()
        self.toolbar = QToolBar("My main toolbar")
        self.addToolBar(self.toolbar)

        self.button_action = QAction("Your button", self)
        self.button_action.setStatusTip("This is your button")
        self.button_action.triggered.connect(self.toolbar_button_clicked)
        self.button_action.setCheckable(True)
        self.toolbar.addAction(self.button_action)


        self.setStatusBar(QStatusBar(self))




    def toolbar_button_clicked(self, s):
        print("click", s)




def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()