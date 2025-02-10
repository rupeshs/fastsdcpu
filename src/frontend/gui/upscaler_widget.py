from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSlider,
    QTabWidget,
    QSpacerItem,
    QSizePolicy,
    QComboBox,
    QCheckBox,
    QTextEdit,
    QToolButton,
    QFileDialog,
    QApplication,
    QRadioButton,
    QFrame,
)
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QDesktopServices, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import QSize, QThreadPool, Qt, QUrl, QBuffer

import io
from PIL import Image
from constants import DEVICE
from PIL.ImageQt import ImageQt
from app_settings import AppSettings
from urllib.parse import urlparse, unquote
from backend.models.upscale import UpscaleMode
from backend.upscale.upscaler import upscale_image
from frontend.gui.img2img_widget import Img2ImgWidget
from paths import FastStableDiffusionPaths, join_paths
from backend.models.lcmdiffusion_setting import DiffusionTask
from frontend.gui.image_generator_worker import ImageGeneratorWorker
from frontend.webui.image_variations_ui import generate_image_variations


class UpscalerWidget(Img2ImgWidget):
    scale_factor = 2.0

    def __init__(self, config: AppSettings, parent):
        super().__init__(config, parent)
        # Hide prompt and negative prompt widgets
        self.prompt.hide()
        self.neg_prompt.hide()
        # self.neg_prompt.deleteLater()
        self.strength_label.hide()
        self.strength.hide()
        self.generate.setText("Upscale")
        # Create upscaler widgets
        self.edsr_button = QRadioButton("EDSR", self)
        self.edsr_button.toggled.connect(self.on_mode_change)
        self.edsr_button.toggle()
        self.sd_button = QRadioButton("SD", self)
        self.sd_button.toggled.connect(self.on_mode_change)
        self.aura_button = QRadioButton("AURA-SR", self)
        self.aura_button.toggled.connect(self.on_mode_change)

        self.neg_prompt_label.setText("Upscale mode (2x) | AURA-SR (4x):")
        # Create upscaler buttons layout
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.edsr_button)
        hlayout.addWidget(self.sd_button)
        hlayout.addWidget(self.aura_button)
        # Can't use a layout with replaceWidget(), so the layout is assigned
        # to a dummy widget used to replace the negative prompt button and
        # obtain the desired GUI design
        self.container = QWidget()
        self.container.setLayout(hlayout)
        self.layout().replaceWidget(self.neg_prompt, self.container)

    def generate_image(self):
        self.parent.prepare_generation_settings(self.config)
        self.config.settings.lcm_diffusion_setting.init_image = Image.open(
            self.img_path.text()
        )
        self.config.settings.lcm_diffusion_setting.strength = self.strength.value() / 10
        upscaled_filepath = FastStableDiffusionPaths.get_upscale_filepath(
            None,
            self.scale_factor,
            self.config.settings.generated_images.format,
        )

        images = upscale_image(
            context=self.parent.context,
            src_image_path=self.img_path.text(),
            dst_image_path=upscaled_filepath,
            upscale_mode=self.upscale_mode,
            strength=0.1,
        )
        self.prepare_images(images)
        self.after_generation()

    def before_generation(self):
        super().before_generation()
        self.container.setEnabled(False)

    def after_generation(self):
        super().after_generation()
        self.container.setEnabled(True)
        # TODO For some reason, if init_image is not set to None, saving settings
        # when closing the main window will fail, even though the save() method
        # explicitly sets init_image to None?
        self.config.settings.lcm_diffusion_setting.init_image = None

    def on_mode_change(self):
        self.scale_factor = 2.0
        if self.edsr_button.isChecked():
            self.upscale_mode = UpscaleMode.normal.value
        elif self.sd_button.isChecked():
            self.upscale_mode = UpscaleMode.sd_upscale.value
        else:
            self.upscale_mode = UpscaleMode.aura_sr.value
            self.scale_factor = 4.0


# Test the widget
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = ImageVariationsWidget(None, None)
    widget.show()
    app.exec()
