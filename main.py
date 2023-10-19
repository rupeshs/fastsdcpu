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

RESULTS_DIRECTORY = "results"


def get_results_path():
    app_dir = os.path.dirname(__file__)
    # work_dir = os.path.dirname(app_dir)
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
        self.pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main",
        )
        self.pipe.to(torch_device="cpu", torch_dtype=torch.float32)
        self.threadpool = QThreadPool()
        self.output_path = get_results_path()
        print(f"Output path : { self.output_path}")

    def init_ui(self):
        self.img = QLabel("<<Image>>")
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setFixedSize(QSize(512, 512))

        self.generate = QPushButton("Generate")
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("Fantasy landscape")

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.prompt)
        hlayout.addWidget(self.generate)

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.img)
        vlayout.addLayout(hlayout)

        widget = QWidget()
        widget.setLayout(vlayout)

        self.setCentralWidget(widget)
        self.generate.clicked.connect(self.text_to_image)

    def generate_image(self):
        print(self.prompt.text())
        tick = time()
        num_inference_steps = 4
        images = self.pipe(
            prompt=self.prompt.text(),
            num_inference_steps=num_inference_steps,
            guidance_scale=8.0,
            lcm_origin_steps=50,
            width=512,
            height=512,
            output_type="pil",
        ).images
        elasped = time() - tick
        print(f"Elasped time : {elasped:.2f} sec")
        image_id = uuid4()
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        images[0].save(os.path.join(self.output_path, f"{image_id}.png"))
        print(f"Images saved {image_id}.png")
        im = ImageQt(images[0]).copy()
        pixmap = QPixmap.fromImage(im)
        self.img.setPixmap(pixmap)

    def text_to_image(self):
        self.img.setText("Please wait...")
        worker = Worker(self.generate_image)
        self.threadpool.start(worker)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
