from time import time
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
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import (
    QSize,
    QThreadPool,
    Qt,
)

from PIL.ImageQt import ImageQt
import os
from uuid import uuid4
from backend.lcm_text_to_image import LCMTextToImage
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
from pprint import pprint
from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO, APP_NAME
from frontend.gui.image_generator_worker import ImageGeneratorWorker
from app_settings import AppSettings
from models.settings import Settings
from paths import FastStableDiffusionPaths


class MainWindow(QMainWindow):
    def __init__(self, app_settings: AppSettings):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setFixedSize(QSize(530, 600))
        self.init_ui()
        self.pipeline = None
        self.threadpool = QThreadPool()
        self.app_settings = app_settings
        self.device = "cpu"
        self.settings = self.app_settings.get_settings()
        self.output_path = self.settings.results_path
        self.seed_value.setEnabled(self.settings.lcm_setting.use_seed)
        self.previous_width = 0
        self.previous_height = 0
        self.lcm_model.setEnabled(not self.settings.lcm_setting.use_openvino)
        self.use_openvino_check.setChecked(self.settings.lcm_setting.use_openvino)
        self.use_openvino = self.use_openvino_check.isChecked()
        self.previous_model = ""
        self.lcm_text_to_image = LCMTextToImage()
        print(f"Output path : { self.output_path}")

    def init_ui(self):
        self.create_main_tab()
        self.create_settings_tab()
        self.create_about_tab()
        self.show()

    def create_main_tab(self):
        self.img = QLabel("<<Image>>")
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setFixedSize(QSize(512, 512))

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("A fantasy landscape")
        self.generate = QPushButton("Generate")
        self.generate.clicked.connect(self.text_to_image)
        self.prompt.setFixedHeight(35)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.prompt)
        hlayout.addWidget(self.generate)

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.img)
        vlayout.addLayout(hlayout)

        self.tab_widget = QTabWidget(self)
        self.tab_main = QWidget()
        self.tab_settings = QWidget()
        self.tab_about = QWidget()
        self.tab_main.setLayout(vlayout)

        self.tab_widget.addTab(self.tab_main, "Text to Image")
        self.tab_widget.addTab(self.tab_settings, "Settings")
        self.tab_widget.addTab(self.tab_about, "About")

        self.setCentralWidget(self.tab_widget)
        self.use_seed = False

    def create_settings_tab(self):
        model_hlayout = QHBoxLayout()
        self.lcm_model_label = QLabel("Latent Consistency Model:")
        self.lcm_model = QLineEdit(LCM_DEFAULT_MODEL)
        model_hlayout.addWidget(self.lcm_model_label)
        model_hlayout.addWidget(self.lcm_model)

        self.inference_steps_value = QLabel("Number of inference steps: 4")
        self.inference_steps = QSlider(orientation=Qt.Orientation.Horizontal)
        self.inference_steps.setMaximum(25)
        self.inference_steps.setMinimum(1)
        self.inference_steps.setValue(4)
        self.inference_steps.valueChanged.connect(self.update_label)

        self.guidance_value = QLabel("Guidance scale: 8")
        self.guidance = QSlider(orientation=Qt.Orientation.Horizontal)
        self.guidance.setMaximum(200)
        self.guidance.setMinimum(10)
        self.guidance.setValue(80)
        self.guidance.valueChanged.connect(self.update_guidance_label)

        self.width_value = QLabel("Width :")
        self.width = QComboBox(self)
        self.width.addItem("256")
        self.width.addItem("512")
        self.width.addItem("768")
        self.width.setCurrentText("512")

        self.height_value = QLabel("Height :")
        self.height = QComboBox(self)
        self.height.addItem("256")
        self.height.addItem("512")
        self.height.addItem("768")
        self.height.setCurrentText("512")

        self.seed_check = QCheckBox("Use seed")
        self.seed_check.stateChanged.connect(self.seed_changed)
        self.seed_value = QLineEdit()
        self.seed_value.setInputMask("9999999999")
        self.seed_value.setText("123123")

        self.safety_checker = QCheckBox("Use safety checker")
        self.safety_checker.setChecked(True)
        self.use_openvino_check = QCheckBox("Use OpenVINO")
        self.use_openvino_check.setChecked(False)
        self.use_local_model_folder = QCheckBox(
            "Use locally cached model or downloaded model folder(offline)"
        )
        self.use_local_model_folder.setChecked(False)
        self.use_openvino_check.stateChanged.connect(self.use_openvino_changed)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.seed_check)
        hlayout.addWidget(self.seed_value)
        hspacer = QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        slider_hspacer = QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)

        vlayout = QVBoxLayout()
        vspacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        vlayout.addItem(hspacer)
        vlayout.addLayout(model_hlayout)
        vlayout.addWidget(self.use_local_model_folder)
        vlayout.addItem(slider_hspacer)
        vlayout.addWidget(self.inference_steps_value)
        vlayout.addWidget(self.inference_steps)
        vlayout.addWidget(self.width_value)
        vlayout.addWidget(self.width)
        vlayout.addWidget(self.height_value)
        vlayout.addWidget(self.height)
        vlayout.addWidget(self.guidance_value)
        vlayout.addWidget(self.guidance)
        vlayout.addLayout(hlayout)
        vlayout.addWidget(self.safety_checker)
        vlayout.addWidget(self.use_openvino_check)
        vlayout.addItem(vspacer)
        self.tab_settings.setLayout(vlayout)

    def create_about_tab(self):
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText(
            """<h1>FastSD CPU v1.0.0 beta 4</h1> 
               <h3>(c)2023 - Rupesh Sreeraman</h3>
                <h3>Faster stable diffusion on CPU</h3>
                 <h3>Based on Latent Consistency Models</h3>
                <h3>GitHub : https://github.com/rupeshs/fastsdcpu/</h3>"""
        )

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.label)
        self.tab_about.setLayout(vlayout)

    def use_openvino_changed(self, state):
        if state == 2:
            self.use_openvino = True
            self.lcm_model.setEnabled(False)
        else:
            self.use_openvino = False
            self.lcm_model.setEnabled(True)

    def update_label(self, value):
        self.inference_steps_value.setText(f"Number of inference steps: {value}")

    def update_guidance_label(self, value):
        val = round(int(value) / 10, 1)
        self.guidance_value.setText(f"Guidance scale: {val}")

    def seed_changed(self, state):
        print(state)
        if state == 2:
            self.use_seed = True
            self.seed_value.setEnabled(True)
        else:
            self.use_seed = False
            print("false")
            self.seed_value.setEnabled(False)

    def generate_image(self):
        lcm_diffusion_setting = LCMDiffusionSetting(
            prompt=self.prompt.toPlainText(),
            image_height=int(self.width.currentText()),
            image_width=int(self.height.currentText()),
            inference_steps=self.inference_steps.value(),
            guidance_scale=round(int(self.guidance.value()) / 10, 1),
            number_of_images=1,
            seed=int(self.seed_value.text()) if self.use_seed else -1,
            use_offline_model=self.use_local_model_folder.isChecked(),
            use_openvino=self.use_openvino,
            use_safety_checker=self.safety_checker.isChecked(),
        )

        if self.use_openvino:
            model_id = LCM_DEFAULT_MODEL_OPENVINO
        else:
            model_id = self.lcm_model.text()

        if self.pipeline is None or self.previous_model != model_id:
            print(f"Using LCM model {model_id}")
            self.lcm_text_to_image.init(
                model_id=model_id,
                use_openvino=self.use_openvino,
                use_local_model=lcm_diffusion_setting.use_offline_model,
            )

        pprint(dict(lcm_diffusion_setting))
        tick = time()
        reshape_required = False
        if self.use_openvino:
            # Dimension changed so reshape and compile
            if (
                self.previous_width != lcm_diffusion_setting.image_width
                or self.previous_height != lcm_diffusion_setting.image_height
                or self.previous_model != model_id
            ):
                pprint("Reshape and compile")
                reshape_required = True

        images = self.lcm_text_to_image.generate(
            lcm_diffusion_setting,
            reshape_required,
        )
        elapsed = time() - tick
        print(f"Elapsed time : {elapsed:.2f} sec")
        image_id = uuid4()
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        images[0].save(os.path.join(self.output_path, f"{image_id}.png"))
        print(f"Image {image_id}.png saved")
        im = ImageQt(images[0]).copy()
        pixmap = QPixmap.fromImage(im)
        self.img.setPixmap(pixmap)
        self.previous_width = lcm_diffusion_setting.image_width
        self.previous_height = lcm_diffusion_setting.image_height
        self.previous_model = model_id

    def text_to_image(self):
        self.img.setText("Please wait...")
        worker = ImageGeneratorWorker(self.generate_image)
        self.threadpool.start(worker)

    def closeEvent(self, event):
        print(self.use_openvino)
        lcm_diffusion_setting = LCMDiffusionSetting(
            prompt=self.prompt.toPlainText(),
            image_height=int(self.width.currentText()),
            image_width=int(self.height.currentText()),
            inference_steps=self.inference_steps.value(),
            guidance_scale=round(int(self.guidance.value()) / 10, 1),
            number_of_images=1,
            seed=int(self.seed_value.text()) if self.use_seed else -1,
            use_offline_model=self.use_local_model_folder.isChecked(),
            use_openvino=self.use_openvino_check.isChecked(),
            use_safety_checker=self.safety_checker.isChecked(),
        )
        app_settings = Settings(
            results_path=FastStableDiffusionPaths().get_results_path(),
            lcm_diffusion_setting=lcm_diffusion_setting,
        )
        print(app_settings.dict())
        self.app_settings.save(app_settings)
