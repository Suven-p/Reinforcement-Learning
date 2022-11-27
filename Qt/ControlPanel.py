from PyQt5 import QtWidgets as qtw, QtCore as qtc


class ControlPanel(qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = qtw.QVBoxLayout()
        self.text1 = qtw.QLabel(
            "Control Panel " + str(self.geometry().width()))
        self.text1.setAlignment(qtc.Qt.AlignCenter)
        self.layout.addWidget(self.text1)
        self.setLayout(self.layout)
