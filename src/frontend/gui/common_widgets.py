from os import path
from PIL import Image
from urllib.parse import urlparse, unquote
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QSlider,
    QLabel,
    QFrame,
    QComboBox,
    QCheckBox,
    QWidget,
    QSizePolicy,
    QMessageBox,
)
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QDesktopServices, QDragEnterEvent, QDropEvent


class LabeledSlider(QWidget):
    """A simple QSlider and QLabel combo for displaying the slider value;
    it holds both an _int_ and _float_ representation of the slider value
    for convenience, the value to display is determined by the
    _use_float_value_ initialization argument.
    """

    valueChanged = QtCore.pyqtSignal(int, float)

    def __init__(self, use_float_value=False):
        super().__init__()
        self.use_float_value = use_float_value
        self._label = QLabel("0")
        if use_float_value:
            self._label.setText("0.00")
        self._slider = QSlider(orientation=Qt.Orientation.Horizontal)
        self._slider.setMaximum(20)
        self._slider.setMinimum(0)
        self._slider.setValue(0)
        self._slider.setTickInterval(1)
        self._slider.setSingleStep(1)
        self._slider.valueChanged.connect(self.onSliderChanged)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self._slider)
        hlayout.addWidget(self._label)
        self.setLayout(hlayout)

    def onSliderChanged(self, value):
        if self.use_float_value:
            self._label.setText("%.2f" % (value / 20.0))
        else:
            self._label.setText("%d" % (value))
        self.valueChanged.emit(value, value / 20.0)

    def setValue(self, value):
        if self.use_float_value:
            self._slider.setValue(int(value * 20))
        else:
            self._slider.setValue(value)

    def getValue(self):
        if self.use_float_value:
            return self._slider.value() / 20.0
        else:
            return self._slider.value()


class ImageLabel(QLabel):
    """A simple QLabel widget for displaying an image; the label image
    can be updated either by calling the _show_image()_ method or by
    dragging an image into the widget area if the label is set to
    accept drag and drop, which is set to _False_ by default.
    """

    changed = QtCore.pyqtSignal()

    def __init__(self, text: str, width=512, height=512):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.resize(width, height)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.sizeHint = QSize(width, height)
        self.width = width
        self.height = height
        self.setAcceptDrops(False)

    def show_image(self, pixmap: QPixmap = None):
        """Updates the widget pixamp"""
        if pixmap == None or pixmap.isNull():
            return
        self.current_pixmap = pixmap
        self.changed.emit()

        # Resize the pixmap to the widget dimensions
        image_width = self.current_pixmap.width()
        image_height = self.current_pixmap.height()
        if image_width > self.width or image_height > self.height:
            new_width = self.width if image_width > self.width else image_width
            new_height = self.height if image_height > self.height else image_height
            self.setPixmap(
                self.current_pixmap.scaled(
                    new_width,
                    new_height,
                    Qt.KeepAspectRatio,
                )
            )
        else:
            self.setPixmap(self.current_pixmap)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasFormat("text/plain"):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        event.acceptProposedAction()
        self.path = unquote(urlparse(event.mimeData().text()).path)
        pixmap = QPixmap(self.path)
        self.show_image(pixmap)
        self.update()


# Test the widget
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.resize(640, 480)
    window.layout().addWidget(LabeledSlider(True))
    window.layout().addWidget(ImageLabel("Hello world!", 128, 128))
    window.show()
    app.exec()
