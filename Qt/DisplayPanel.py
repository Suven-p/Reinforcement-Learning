from PyQt5 import QtWidgets as qtw, QtCore as qtc


class DisplayPanel(qtw.QWidget):
    def __init__(self, size, parent=None):
        super().__init__(parent)
        self.layout = qtw.QGridLayout()
        for row in range(size[0]):
            for col in range(size[0]):
                box = qtw.QWidget()
                box.setStyleSheet(
                    "background-color: #f029be; border: 2px solid blue;")
                label = qtw.QLabel(f"{row},{col}")
                label.setStyleSheet("border: 0")
                label.setAlignment(qtc.Qt.AlignTop | qtc.Qt.AlignHCenter)
                layout = qtw.QVBoxLayout()
                layout.addWidget(label)
                box.setLayout(layout)
                self.layout.addWidget(box, row, col)
        self.setLayout(self.layout)
