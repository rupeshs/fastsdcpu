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
from frontend.gui.img2img_widget import Img2ImgWidget
from backend.models.lcmdiffusion_setting import DiffusionTask
from frontend.gui.image_generator_worker import ImageGeneratorWorker
from frontend.webui.image_variations_ui import generate_image_variations


class ImageVariationsWidget(Img2ImgWidget):
    def __init__(self, config: AppSettings, parent):
        super().__init__(config, parent)
        # Hide prompt and negative prompt widgets
        self.prompt.hide()
        self.neg_prompt_label.hide()
        self.neg_prompt.setEnabled(False)

    def generate_image(self):
        self.parent.prepare_generation_settings(self.config)
        self.config.settings.lcm_diffusion_setting.diffusion_task = (
            DiffusionTask.image_to_image.value
        )
        self.config.settings.lcm_diffusion_setting.prompt = ""
        self.config.settings.lcm_diffusion_setting.negative_prompt = ""
        self.config.settings.lcm_diffusion_setting.init_image = Image.open(
            self.img_path.text()
        )
        self.config.settings.lcm_diffusion_setting.strength = self.strength.value() / 10

        images = generate_image_variations(
            self.config.settings.lcm_diffusion_setting.init_image,
            self.config.settings.lcm_diffusion_setting.strength,
        )
        self.prepare_images(images)
        self.after_generation()


# Test the widget
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = ImageVariationsWidget(None, None)
    widget.show()
    app.exec()
