from diffusers import DiffusionPipeline
import torch
from time import time
import sys
from PyQt5.QtWidgets import (
    QApplication,
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
    pyqtSignal,
    pyqtSlot,
    QObject,
    QRunnable,
    QThreadPool,
    Qt,
)
from PIL.ImageQt import ImageQt
import traceback, sys
import os
from uuid import uuid4
from src.backend.lcmdiffusion.pipelines.openvino.lcm_ov_pipeline import (
    OVLatentConsistencyModelPipeline,
)
from src.backend.lcmdiffusion.pipelines.openvino.lcm_scheduler import LCMScheduler
import numpy as np


RESULTS_DIRECTORY = "results"


def get_lcm_diffusion_pipeline_path():
    main_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        main_path,
        "src",
        "backend",
        "lcmdiffusion",
        "pipelines",
        "latent_consistency_txt2img.py",
    )
    return file_path


def get_results_path():
    app_dir = os.path.dirname(__file__)
    config_path = os.path.join(app_dir, RESULTS_DIRECTORY)
    return config_path


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FastSD CPU")
        self.setFixedSize(QSize(530, 600))
        self.init_ui()
        self.pipeline = None
        self.threadpool = QThreadPool()
        self.output_path = get_results_path()
        self.device = "cpu"
        self.seed_value.setEnabled(False)
        self.previous_width = 0
        self.previous_height = 0
        print(f"Output path : { self.output_path}")
        self.lcm_model.setEnabled(True)
        self.use_openvino_check.setChecked(False)
        self.use_openvino = self.use_openvino_check.isChecked()
        self.previous_model = ""

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
        self.lcm_model = QLineEdit("SimianLuo/LCM_Dreamshaper_v7")
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
        if state == 2:
            self.use_seed = True
            self.seed_value.setEnabled(True)
        else:
            self.use_seed = False
            self.seed_value.setEnabled(False)

    def generate_image(self):
        prompt = self.prompt.toPlainText()
        guidance_scale = round(int(self.guidance.value()) / 10, 1)
        img_width = int(self.width.currentText())
        img_height = int(self.height.currentText())
        num_inference_steps = self.inference_steps.value()
        if self.use_openvino:
            model_id = "deinferno/LCM_Dreamshaper_v7-openvino"
        else:
            model_id = self.lcm_model.text()

        if self.pipeline is None or self.previous_model != model_id:
            print(f"Using LCM model {model_id}")
            if self.use_openvino:
                if self.pipeline:
                    del self.pipeline
                scheduler = LCMScheduler.from_pretrained(
                    model_id,
                    subfolder="scheduler",
                )
                self.pipeline = OVLatentConsistencyModelPipeline.from_pretrained(
                    "deinferno/LCM_Dreamshaper_v7-openvino",
                    scheduler=scheduler,
                    compile=False,
                    local_files_only=self.use_local_model_folder.isChecked(),
                )
            else:
                if self.pipeline:
                    del self.pipeline
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    custom_pipeline=get_lcm_diffusion_pipeline_path(),
                    custom_revision="main",
                    local_files_only=self.use_local_model_folder.isChecked(),
                )
                self.pipeline.to(
                    torch_device=self.device,
                    torch_dtype=torch.float32,
                )

        if self.use_seed:
            cur_seed = int(self.seed_value.text())
            if self.use_openvino:
                np.random.seed(cur_seed)
            else:
                torch.manual_seed(cur_seed)

        print(f"Prompt : {prompt}")
        print(f"Resolution : {img_width} x {img_height}")
        print(f"Guidance Scale : {guidance_scale}")
        print(f"Inference_steps  : {num_inference_steps}")
        if self.use_seed:
            print(f"Seed: {cur_seed}")

        tick = time()

        if self.use_openvino:
            print("Using OpenVINO")
            # Dimension changed so reshape and compile
            if (
                self.previous_width != img_width
                or self.previous_height != img_height
                or self.previous_model != model_id
            ):
                print("Reshape and compile")
                self.pipeline.reshape(
                    batch_size=1,
                    height=img_height,
                    width=img_width,
                    num_images_per_prompt=1,
                )
                self.pipeline.compile()

        if not self.safety_checker.isChecked():
            self.pipeline.safety_checker = None
        images = self.pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            lcm_origin_steps=50,
            width=img_width,
            height=img_height,
            output_type="pil",
        ).images
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
        self.previous_width = img_width
        self.previous_height = img_height
        self.previous_model = model_id

    def text_to_image(self):
        self.img.setText("Please wait...")
        worker = Worker(self.generate_image)
        self.threadpool.start(worker)

    def latents_callback(self, i, t, latents):
        print(i)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
