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
from frontend.gui.base_widget import BaseWidget
from backend.models.lcmdiffusion_setting import DiffusionTask
from frontend.gui.image_generator_worker import ImageGeneratorWorker


class Img2ImgWidget(BaseWidget):
    def __init__(self, config: AppSettings, parent):
        super().__init__(config)
        self.config = config
        self.parent = parent
        self.generate.clicked.connect(self.img2img_click)

        self.strength_label = QLabel("Denoising strength: 0.3")
        self.strength = QSlider(orientation=Qt.Orientation.Horizontal)
        self.strength.setMaximum(10)
        self.strength.setMinimum(1)
        self.strength.setValue(3)
        self.strength.valueChanged.connect(self.update_strength_label)
        # self.layout().insertWidget(1, self.strength_label)
        # self.layout().insertWidget(2, self.strength)
        self.layout().addWidget(self.strength_label)
        self.layout().addWidget(self.strength)


    def img2img_click(self):
        self.img.setText("Please wait...")
        worker = ImageGeneratorWorker(self.generate_image)
        self.parent.threadpool.start(worker)

    def generate_image(self):
        self.parent.prepare_generation_settings(self.config)
        self.config.settings.lcm_diffusion_setting.diffusion_task = DiffusionTask.image_to_image.value
        self.config.settings.lcm_diffusion_setting.prompt = self.prompt.toPlainText()
        self.config.settings.lcm_diffusion_setting.negative_prompt = (
            self.neg_prompt.toPlainText()
        )
        self.config.settings.lcm_diffusion_setting.init_image = self.image_from_pixmap(self.pixmap)

        images = self.parent.context.generate_text_to_image(
            self.config.settings,
            self.config.reshape_required,
            DEVICE,
        )
        self.prepare_images(images)

        # TODO Is it possible to move the next lines to a separate function?
        self.parent.previous_width = self.config.settings.lcm_diffusion_setting.image_width
        self.parent.previous_height = self.config.settings.lcm_diffusion_setting.image_height
        self.parent.previous_model = self.config.model_id
        self.parent.previous_num_of_images = (
            self.config.settings.lcm_diffusion_setting.number_of_images
        )

    def update_strength_label(self, value):
        val = round(int(value) / 10, 1)
        self.strength_label.setText(f"Denoising strength: {val}")
        self.config.settings.lcm_diffusion_setting.strength = val


# Test the widget
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = Img2ImgWidget(None, None)
    widget.show()
    app.exec()

