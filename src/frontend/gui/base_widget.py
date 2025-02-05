from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QLayout,
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
from PyQt5.QtGui import (
    QPixmap,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QMouseEvent,
)
from PyQt5.QtCore import QSize, QThreadPool, Qt, QUrl, QBuffer

import io
from PIL import Image
from constants import DEVICE
from PIL.ImageQt import ImageQt
from app_settings import AppSettings
from urllib.parse import urlparse, unquote
from frontend.gui.image_generator_worker import ImageGeneratorWorker


class ImageLabel(QLabel):
    """Defines a simple QLabel widget"""

    changed = QtCore.pyqtSignal()

    def __init__(self, text: str):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.resize(512, 512)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.sizeHint = QSize(512, 512)
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
        if image_width > 512 or image_height > 512:
            new_width = 512 if image_width > 512 else image_width
            new_height = 512 if image_height > 512 else image_height
            self.setPixmap(
                self.current_pixmap.scaled(
                    new_width,
                    new_height,
                    Qt.KeepAspectRatio,
                )
            )
        else:
            self.setPixmap(self.current_pixmap)


class BaseWidget(QWidget):
    def __init__(self, config: AppSettings, parent):
        super().__init__()
        self.config = config
        self.gen_images = []
        self.image_index = 0
        self.config = config
        self.parent = parent

        # Initialize GUI widgets
        self.prev_btn = QToolButton()
        self.prev_btn.setText("<")
        self.prev_btn.clicked.connect(self.on_show_previous_image)
        self.img = ImageLabel("<<Image>>")
        self.next_btn = QToolButton()
        self.next_btn.setText(">")
        self.next_btn.clicked.connect(self.on_show_next_image)
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("A fantasy landscape")
        self.prompt.setAcceptRichText(False)
        self.prompt.setFixedHeight(40)
        self.neg_prompt = QTextEdit()
        self.neg_prompt.setPlaceholderText("")
        self.neg_prompt.setAcceptRichText(False)
        self.neg_prompt_label = QLabel("Negative prompt (Set guidance scale > 1.0):")
        self.neg_prompt.setFixedHeight(35)
        self.neg_prompt.setEnabled(False)
        self.generate = QPushButton("Generate")
        self.generate.clicked.connect(self.generate_click)
        self.browse_results = QPushButton("...")
        self.browse_results.setFixedWidth(30)
        self.browse_results.clicked.connect(self.on_open_results_folder)
        self.browse_results.setToolTip("Open output folder")

        # Create the image navigation layout
        ilayout = QHBoxLayout()
        ilayout.addWidget(self.prev_btn)
        ilayout.addWidget(self.img)
        ilayout.addWidget(self.next_btn)

        # Create the generate button layout
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.neg_prompt)
        hlayout.addWidget(self.generate)
        hlayout.addWidget(self.browse_results)

        # Create the actual widget layout
        vlayout = QVBoxLayout()
        vlayout.addLayout(ilayout)
        # vlayout.addItem(self.vspacer)
        vlayout.addWidget(self.prompt)
        vlayout.addWidget(self.neg_prompt_label)
        vlayout.addLayout(hlayout)
        self.setLayout(vlayout)

        self.parent.settings_changed.connect(self.on_settings_changed)

    def generate_image(self):
        self.parent.prepare_generation_settings(self.config)
        self.config.settings.lcm_diffusion_setting.prompt = self.prompt.toPlainText()
        self.config.settings.lcm_diffusion_setting.negative_prompt = (
            self.neg_prompt.toPlainText()
        )
        images = self.parent.context.generate_text_to_image(
            self.config.settings,
            self.config.reshape_required,
            DEVICE,
        )
        self.prepare_images(images)
        self.after_generation()

    def prepare_images(self, images):
        """Prepares the generated images to be displayed in the Qt widget"""
        self.image_index = 0
        self.gen_images = []
        for img in images:
            im = ImageQt(img).copy()
            pixmap = QPixmap.fromImage(im)
            self.gen_images.append(pixmap)

        if len(self.gen_images) > 1:
            self.next_btn.setEnabled(True)
            self.prev_btn.setEnabled(False)
        else:
            self.next_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)

        self.img.show_image(pixmap=self.gen_images[0])

    def on_show_next_image(self):
        if self.image_index != len(self.gen_images) - 1 and len(self.gen_images) > 0:
            self.prev_btn.setEnabled(True)
            self.image_index += 1
            self.img.show_image(pixmap=self.gen_images[self.image_index])
            if self.image_index == len(self.gen_images) - 1:
                self.next_btn.setEnabled(False)

    def on_show_previous_image(self):
        if self.image_index != 0:
            self.next_btn.setEnabled(True)
            self.image_index -= 1
            self.img.show_image(pixmap=self.gen_images[self.image_index])
            if self.image_index == 0:
                self.prev_btn.setEnabled(False)

    def on_open_results_folder(self):
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(self.config.settings.generated_images.path)
        )

    def generate_click(self):
        self.img.setText("Please wait...")
        self.before_generation()
        worker = ImageGeneratorWorker(self.generate_image)
        self.parent.threadpool.start(worker)

    def before_generation(self):
        """Call this function before running a generation task"""
        self.img.setEnabled(False)
        self.generate.setEnabled(False)
        self.browse_results.setEnabled(False)

    def after_generation(self):
        """Call this function after running a generation task"""
        self.img.setEnabled(True)
        self.generate.setEnabled(True)
        self.browse_results.setEnabled(True)
        self.parent.store_dimension_settings()

    def on_settings_changed(self):
        self.neg_prompt.setEnabled(
            self.config.settings.lcm_diffusion_setting.use_openvino
            or self.config.settings.lcm_diffusion_setting.use_lcm_lora
        )


# Test the widget
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = BaseWidget(None)
    widget.show()
    app.exec()
