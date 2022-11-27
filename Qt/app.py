import sys
from typing import List, Tuple
from PyQt5 import QtWidgets as qtw
from qt_material import apply_stylesheet
from ControlPanel import ControlPanel
from DisplayPanel import DisplayPanel
from utils import centerWidget, setStretch


class MainWindow(qtw.QMainWindow):
    def __init__(self, title: str, size: Tuple[int], children: List[qtw.QWidget] = None, parent: qtw.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*size)
        self.centralWidget = qtw.QWidget(self)
        self.centralWidgetLayout = qtw.QHBoxLayout()
        self.addCentralChildren(*children)
        self.centralWidget.setLayout(self.centralWidgetLayout)
        self.setCentralWidget(self.centralWidget)
        centerWidget(self, 0)

    def setCentralLayout(self, layout: qtw.QLayout):
        self.centralWidgetLayout = layout
        self.centralWidget.setLayout(self.centralWidgetLayout)

    def addCentralChildren(self, *children: List[qtw.QWidget]):
        if not children:
            return
        for child in children:
            self.centralWidgetLayout.addWidget(child)


def main():
    windowWidth = 2000
    windowHeight = 1000

    app = qtw.QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_blue.xml')
    app.setStyleSheet(app.styleSheet() + "QLabel{font-size: 12pt;}")

    controlPanel = ControlPanel()
    displayPanel = DisplayPanel((4, 4))
    controlPanelWidth = 40
    setStretch(controlPanel, horizontal=controlPanelWidth)
    setStretch(displayPanel, horizontal=100-controlPanelWidth)

    mainwindow = MainWindow(title="Grid World Visualization",
                            children=[controlPanel, displayPanel],
                            size=(windowWidth, windowHeight))
    mainwindow.show()
    app.exec_()


if __name__ == "__main__":
    main()
