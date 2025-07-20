from os import path
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QSlider,
    QLabel,
    QFrame,
    QComboBox,
    QWidget,
    QSizePolicy,
    QMessageBox,
)
from backend.lora import (
    get_lora_models,
    get_active_lora_weights,
    update_lora_weights,
    load_lora_weight,
)
from app_settings import AppSettings
from paths import FastStableDiffusionPaths
if __name__ != "__main__":
    from state import get_settings, get_context
    from models.interface_types import InterfaceType

    app_settings = get_settings()


_MAX_LORA_WEIGHTS = 5
_current_lora_count = 0
_active_lora_widgets = []


class _LabeledSlider(QWidget):
    def __init__(self):
        super().__init__()
        self._label = QLabel("0.00")
        self._slider = QSlider(orientation = Qt.Orientation.Horizontal)
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
        self._label.setText("%.2f" %(value / 20.0))

    def setValue(self, value: int):
        self._slider.setValue(int(value * 20))

    def getValue(self):
        return self._slider.value() / 20.0


# This is a simple widget for displaying the loaded LoRAs name and weight
class _LoraWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.name_label = QLabel()
        self.strength_slider = _LabeledSlider()
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.name_label)
        hlayout.addWidget(self.strength_slider)
        self.setLayout(hlayout)

    def setValues(self, name: str, weight: float):
        self.name_label.setText(name)
        self.strength_slider.setValue(weight)

    def getValues(self):
        return (self.name_label.text(), self.strength_slider.getValue())


class LoraModelsWidget(QWidget):
    def __init__(self, config: AppSettings, parent):
        super().__init__()
        self.parent = parent
        lora_models_map = {}
        if config != None:
            lora_models_map = get_lora_models(config.settings.lcm_diffusion_setting.lora.models_dir)
        self.models_combobox = QComboBox()
        self.models_combobox.addItems(lora_models_map.keys())
        self.models_combobox.setToolTip("<p style='white-space:pre'>Place LoRA models in the <b>lora_models</b> folder</p>")
        self.weight_slider = _LabeledSlider()
        self.load_button = QPushButton("Load selected LoRA")
        self.load_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.load_button.clicked.connect(self.on_load_lora)
        self.loaded_label = QLabel("Loaded LoRA models:")
        self.update_button = QPushButton("Update LoRA weights")
        self.update_button.clicked.connect(self.on_update_weights)
        self.separator = QLabel()
        self.separator.setFrameShape(QFrame.HLine)

        glayout = QGridLayout()
        glayout.setVerticalSpacing(0)
        glayout.addWidget(QLabel("LoRA model:"), 0, 0)
        glayout.addWidget(QLabel("Initial LoRA weight:", ), 0, 1)
        glayout.addWidget(self.models_combobox, 1, 0)
        glayout.addWidget(self.weight_slider, 1, 1)
        glayout.addWidget(self.load_button, 0, 2, 2, 1)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.loaded_label)
        hlayout.addWidget(self.update_button)
        vlayout = QVBoxLayout()
        vlayout.addLayout(glayout, 10)
        vlayout.addWidget(self.separator, 1)
        vlayout.addLayout(hlayout, 10)
        vlayout.addStretch(80)
        self.setLayout(vlayout)

    def on_load_lora(self):
        # Code for testing the GUI; ignore when running FastSD CPU
        if __name__ == "__main__":
            self.layout().insertWidget(3, _LoraWidget())
            return
        # End of code for testing the GUI
        
        global _current_lora_count
        global _active_lora_widgets
        if app_settings == None or _current_lora_count >= _MAX_LORA_WEIGHTS:
            return
        if app_settings.settings.lcm_diffusion_setting.use_openvino:
            QMessageBox().information(self.parent, "Error", "LoRA suppport is currently not implemented for OpenVINO.")
            return
        lora_models_map = get_lora_models(
            app_settings.settings.lcm_diffusion_setting.lora.models_dir
        )

        # Load a new LoRA
        settings = app_settings.settings.lcm_diffusion_setting
        settings.lora.fuse = False
        settings.lora.enabled = False
        current_lora = self.models_combobox.currentText()
        current_weight = self.weight_slider.getValue()
        print(f"Selected Lora Model :{current_lora}")
        print(f"Lora weight :{current_weight}")
        settings.lora.path = lora_models_map[current_lora]
        settings.lora.weight = current_weight
        if not path.exists(settings.lora.path):
            QMessageBox.information(self.parent, "Error", "Invalid LoRA model path!")
            return
        pipeline = get_context(InterfaceType.GUI).lcm_text_to_image.pipeline
        if not pipeline:
            QMessageBox.information(self.parent, "Error", "Pipeline not initialized. Please generate an image first.")
            return
        settings.lora.enabled = True
        load_lora_weight(
            get_context(InterfaceType.GUI).lcm_text_to_image.pipeline,
            settings,
        )
        lora_widget = _LoraWidget()
        lora_widget.setValues(current_lora, current_weight)
        self.layout().insertWidget(3, lora_widget)
        _active_lora_widgets.append(lora_widget)
        _current_lora_count += 1

    def on_update_weights(self):
        update_weights = []
        active_weights = get_active_lora_weights()
        if not len(active_weights):
            # QMessageBox.information(self.parent, "Error", "No active LoRAs, first you need to load LoRA model")
            return
        global _active_lora_widgets
        for idx, lora in enumerate(active_weights):
            update_weights.append(
                (
                    lora[0],
                    _active_lora_widgets[idx].getValues()[1],
                )
            )
        if len(update_weights) > 0:
            update_lora_weights(
                get_context(InterfaceType.GUI).lcm_text_to_image.pipeline,
                app_settings.settings.lcm_diffusion_setting,
                update_weights,
            )
            print(_active_lora_widgets)


# Test the widget
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = LoraModelsWidget(None, None)
    widget.show()
    app.exec()


