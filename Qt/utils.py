from PyQt5 import QtWidgets as qtw


def setStretch(widget: qtw.QWidget, horizontal: int = None, vertical: int = None):
    sizePolicy = widget.sizePolicy()
    if horizontal is not None:
        sizePolicy.setHorizontalStretch(horizontal)
    if vertical is not None:
        sizePolicy.setVerticalStretch(vertical)
    widget.setSizePolicy(sizePolicy)


def centerWidget(root: qtw.QWidget, screen: int = 0):
    monitor = qtw.QDesktopWidget().screenGeometry(screen)
    left = monitor.left()
    top = monitor.top()
    width = monitor.width()
    height = monitor.height()
    root.move(int(left + width / 2 - root.geometry().width() / 2),
              int(top + height / 2 - root.geometry().height() / 2))
