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
    QToolButton,
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
    QFileDialog,
)
from PyQt5.QtCore import QSize
from app_settings import AppSettings
from PyQt5.QtGui import QPixmap, QDesktopServices, QDragEnterEvent, QDropEvent
from paths import FastStableDiffusionPaths
from backend.models.lcmdiffusion_setting import ControlNetSetting
from backend.annotators.image_control_factory import ImageControlFactory
from frontend.gui.common_widgets import LabeledSlider, ImageLabel

if __name__ != "__main__":
    from state import get_settings, get_context
    from models.interface_types import InterfaceType
    from backend.lora import get_lora_models

    app_settings = get_settings()


_controlnet_models_map = {}
_current_controlnet_image = None
_current_controlnet_weight = 0.0
_current_controlnet_adapter = ""
_current_controlnet_enabled = False


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
        if len(_controlnet_models_map) > 0:
            _current_controlnet_adapter = list(_controlnet_models_map.keys())[0]
        self.message_label = QLabel(
            "<p style='white-space:pre'>Download ControlNet v1.1 models (723 MB files) from "
            "<a href='https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/tree/main'>"
            "ControlNet v1.1</a>.<br>Place the models in the <b>controlnet_models</b> folder."
            "<br>Restart the application to detect newly installed models.</p>"
        )
        self.message_label.setOpenExternalLinks(True)
        self.enabled_checkbox = QCheckBox("Enable ControlNet")
        self.enabled_checkbox.setEnabled(False)
        self.enabled_checkbox.stateChanged.connect(self.on_enable_changed)
        if len(_controlnet_models_map) > 0:
            self.enabled_checkbox.setEnabled(True)
        self.models_combobox = QComboBox()
        self.models_combobox.setMaximumWidth(160)
        self.models_combobox.addItems(_controlnet_models_map.keys())
        self.models_combobox.setToolTip(
            "<p style='white-space:pre'>Place ControlNet models in the <b>controlnet_models</b> folder</p>"
        )
        self.models_combobox.currentTextChanged.connect(self.on_combo_changed)
        self.weight_slider = LabeledSlider(True)
        self.weight_slider.setValue(0.5)
        self.weight_slider.valueChanged.connect(self.on_weight_changed)
        self.image_button = QToolButton()
        self.image_button.setText("...")
        self.image_button.setToolTip("Click to select control image.")
        self.image_button.clicked.connect(self.controlnet_file_dialog)
        self.image_label = ImageLabel("<<Control image>>", 256, 256)
        self.image_label.setMinimumSize(QSize(256, 256))
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setAcceptDrops(True)
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

        self.container = QFrame()
        self.container.setEnabled(False)
        self.separator = QLabel()
        self.separator.setFrameShape(QFrame.HLine)
        hlayout = QGridLayout(self.container)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addWidget(QLabel("ControlNet model:"), 0, 0)
        hlayout.addWidget(QLabel("Conditioning scale:"), 0, 1)
        hlayout.addWidget(QLabel("Control image:"), 0, 2)
        hlayout.addWidget(self.models_combobox, 1, 0, Qt.AlignTop)
        hlayout.addWidget(self.weight_slider, 1, 1, Qt.AlignTop)
        hlayout.addWidget(self.image_label, 1, 2)
        hlayout.addWidget(self.image_button, 1, 3, Qt.AlignTop)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.message_label, 3)
        vlayout.addWidget(self.enabled_checkbox, 3)
        vlayout.addWidget(self.separator, 1)
        vlayout.addWidget(self.container)
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
        self.container.setEnabled(_current_controlnet_enabled)
        self.update_controlnet_settings()

    def on_combo_changed(self, text: str):
        global _current_controlnet_adapter
        _current_controlnet_adapter = text
        self.update_controlnet_settings()

    def on_weight_changed(self, value: int, valuef: float):
        global _current_controlnet_weight
        _current_controlnet_weight = valuef
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
        settings = app_settings.settings.lcm_diffusion_setting
        if settings.controlnet is None:
            settings.controlnet = ControlNetSetting()
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

    def controlnet_file_dialog(self):
        fileName = QFileDialog.getOpenFileName(
            self, "Open Image", "results", "Image Files (*.png *.jpg *.bmp *.webp)"
        )
        if fileName[0] != "":
            self.image_label.path = fileName[0]
            pixmap = QPixmap(fileName[0])
            self.image_label.show_image(pixmap)
            self.image_label.update()


# Test the widget
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = ControlNetWidget(None, None)
    widget.show()
    app.exec()
