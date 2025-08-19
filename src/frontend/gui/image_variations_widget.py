from PIL import Image
from constants import DEVICE
from PyQt5.QtWidgets import QApplication

from app_settings import AppSettings
from backend.models.lcmdiffusion_setting import DiffusionTask
from frontend.gui.img2img_widget import Img2ImgWidget
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

        images = self.parent.context.generate_text_to_image(
            self.config.settings,
            self.config.reshape_required,
            DEVICE,
        )
        self.prepare_images(images)
        self.after_generation()
