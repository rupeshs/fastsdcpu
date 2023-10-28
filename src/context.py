from typing import Any
from app_settings import Settings
from models.interface_types import InterfaceType
from backend.lcm_text_to_image import LCMTextToImage
from time import time
from backend.image_saver import ImageSaver


class Context:
    def __init__(
        self,
        interface_type: InterfaceType,
        device="cpu",
    ):
        self.interface_type = interface_type
        self.lcm_text_to_image = LCMTextToImage(device)

    def generate_text_to_image(
        self,
        settings: Settings,
        reshape: bool = False,
        device: str = "cpu",
    ) -> Any:
        tick = time()
        self.lcm_text_to_image.init(
            settings.lcm_diffusion_setting.lcm_model_id,
            settings.lcm_diffusion_setting.use_openvino,
            device,
            settings.lcm_diffusion_setting.use_offline_model,
        )
        images = self.lcm_text_to_image.generate(
            settings.lcm_diffusion_setting,
            reshape,
        )
        elapsed = time() - tick
        ImageSaver.save_images(
            settings.results_path,
            images=images,
            lcm_diffusion_setting=settings.lcm_diffusion_setting,
        )
        print(f"Elapsed time : {elapsed:.2f} sec")
        return images
