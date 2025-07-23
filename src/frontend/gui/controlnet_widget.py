from os import path
from PIL import Image
from urllib.parse import urlparse, unquote
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
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
from app_settings import AppSettings
from PyQt5.QtGui import QPixmap, QDesktopServices, QDragEnterEvent, QDropEvent
from paths import FastStableDiffusionPaths
from backend.models.lcmdiffusion_setting import ControlNetSetting
from backend.annotators.image_control_factory import ImageControlFactory

if __name__ != "__main__":
    from state import get_settings, get_context
    from models.interface_types import InterfaceType
    from backend.lora import get_lora_models

    # from frontend.gui.lora_widget import (
    #    _LabeledSlider,
    # )

    app_settings = get_settings()


_controlnet_models_map = {}
_current_controlnet_image = None
_current_controlnet_weight = 0.0
_current_controlnet_adapter = ""
_current_controlnet_enabled = False


# This class can be merged with BaseWidget.ImageLabel
class ImageLabelDrop(QLabel):
    changed = QtCore.pyqtSignal()

    def __init__(self, text: str):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.resize(512, 512)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.sizeHint = QSize(512, 512)
        self.setAcceptDrops(True)

    def show_image(self, pixmap: QPixmap = None):
        """Updates the widget pixamp"""
        if pixmap == None or pixmap.isNull():
            return
        self.current_pixmap = pixmap
        self.changed.emit()

        # Resize the pixmap to the widget dimensions
        image_width = self.current_pixmap.width()
        image_height = self.current_pixmap.height()
        if image_width > 256 or image_height > 256:
            new_width = 256 if image_width > 256 else image_width
            new_height = 256 if image_height > 256 else image_height
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


class ControlNetWidget(QWidget):
    def __init__(self, config: AppSettings, parent):
        super().__init__()
        self.parent = parent
        global _controlnet_models_map
        global _current_controlnet_adapter
        _controlnet_models_map = {}
        if config != None:
            _controlnet_models_map = get_lora_models(
                config.settings.lcm_diffusion_setting.dirs["controlnet"]
            )
        _current_controlnet_adapter = list(_controlnet_models_map.keys())[0]
        self.enabled_checkbox = QCheckBox("Enable ControlNet")
        self.enabled_checkbox.stateChanged.connect(self.on_enable_changed)
        self.models_combobox = QComboBox()
        self.models_combobox.addItems(_controlnet_models_map.keys())
        self.models_combobox.setToolTip(
            "<p style='white-space:pre'>Place LoRA models in the <b>controlnet_models</b> folder</p>"
        )
        self.models_combobox.setEnabled(False)
        self.models_combobox.currentTextChanged.connect(self.on_combo_changed)
        self.weight_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        self.weight_slider.setMaximum(20)
        self.weight_slider.setEnabled(False)
        self.weight_slider.valueChanged.connect(self.on_weight_changed)
        self.image_label = ImageLabelDrop("Drag and drop control image here")
        self.image_label.resize(256, 256)
        self.image_label.sizeHint = QSize(256, 256)
        self.image_label.setMinimumSize(QSize(256, 256))
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setAcceptDrops(True)
        self.image_label.setEnabled(False)
        self.image_label.changed.connect(self.on_image_changed)

        radio_buttons_layout = QHBoxLayout()
        self.radio_buttons_group = QButtonGroup()
        radio_buttons_text = [
            "Canny",
            "Depth",
            "Lineart",
            "MLSD",
            "NormalBAE",
            "Pose",
            "SoftEdge",
            "Shuffle",
            "None",
        ]
        for text in radio_buttons_text:
            radio_button = QRadioButton(text)
            radio_buttons_layout.addWidget(radio_button)
            self.radio_buttons_group.addButton(radio_button)
        radio_button.setChecked(True)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.models_combobox)
        hlayout.addWidget(self.weight_slider)
        hlayout.addWidget(self.image_label)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.enabled_checkbox, 5)
        vlayout.addLayout(hlayout)
        vlayout.addLayout(radio_buttons_layout)
        vlayout.addStretch(80)
        self.setLayout(vlayout)

    def on_image_changed(self):
        global _current_controlnet_image
        _current_controlnet_image = Image.open(self.image_label.path)
        selected_preprocessor = self.radio_buttons_group.checkedButton().text()
        if selected_preprocessor != "None":
            image_control_factory = ImageControlFactory()
            control = image_control_factory.create_control(selected_preprocessor)
            _current_controlnet_image = control.get_control_image(
                _current_controlnet_image
            )
        self.update_controlnet_settings()

    def on_enable_changed(self, state: int):
        global _current_controlnet_enabled
        _current_controlnet_enabled = False
        if state == Qt.Checked:
            _current_controlnet_enabled = True
        self.models_combobox.setEnabled(_current_controlnet_enabled)
        self.weight_slider.setEnabled(_current_controlnet_enabled)
        self.image_label.setEnabled(_current_controlnet_enabled)
        self.update_controlnet_settings()

    def on_combo_changed(self, text: str):
        global _current_controlnet_adapter
        _current_controlnet_adapter = text
        self.update_controlnet_settings()

    def on_weight_changed(self, value: int):
        global _current_controlnet_weight
        _current_controlnet_weight = float(value / 20.0)
        self.update_controlnet_settings()

    def update_controlnet_settings(self):
        # Code for testing the GUI; ignore when running FastSD CPU
        if __name__ == "__main__":
            return

        global _current_controlnet_enabled
        global _current_controlnet_adapter
        global _current_controlnet_weight
        global _current_controlnet_image
        global _controlnet_models_map
        _current_controlnet_enabled = True
        settings = app_settings.settings.lcm_diffusion_setting
        if not _current_controlnet_enabled:
            settings.controlnet.enabled = False
        else:
            settings.controlnet.enabled = True
            settings.controlnet.adapter_path = _controlnet_models_map[
                _current_controlnet_adapter
            ]
            settings.controlnet.conditioning_scale = _current_controlnet_weight
            settings.controlnet._control_image = _current_controlnet_image
        # Currently, every change made to the ControlNet settings will
        # trigger a pipeline rebuild, this can probably be improved
        settings.rebuild_pipeline = True


# Test the widget
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = ControlNetWidget(None, None)
    widget.show()
    app.exec()
