from typing import Any
from app_settings import Settings
from models.interface_types import InterfaceType
from backend.lcm_text_to_image import LCMTextToImage
from time import perf_counter
from backend.image_saver import ImageSaver
from pprint import pprint


class Context:
    def __init__(
        self,
        interface_type: InterfaceType,
        device="cpu",
    ):
        self.interface_type = interface_type.value
        self.lcm_text_to_image = LCMTextToImage(device)

    def generate_text_to_image(
        self,
        settings: Settings,
        reshape: bool = False,
        device: str = "cpu",
        save_images=True,
        save_config=True,
    ) -> Any:
        if (
            settings.lcm_diffusion_setting.use_tiny_auto_encoder
            and settings.lcm_diffusion_setting.use_openvino
        ):
            print(
                "WARNING: Tiny AutoEncoder is not supported in Image to image mode (OpenVINO)"
            )
        tick = perf_counter()
        from state import get_settings

        if save_config:
            get_settings().save()

        pprint(settings.lcm_diffusion_setting.model_dump())
        if not settings.lcm_diffusion_setting.lcm_lora:
            return None
        self.lcm_text_to_image.init(
            device,
            settings.lcm_diffusion_setting,
        )
        images = self.lcm_text_to_image.generate(
            settings.lcm_diffusion_setting,
            reshape,
        )
        elapsed = perf_counter() - tick

        if save_images:
            ImageSaver.save_images(
                settings.results_path,
                images=images,
                lcm_diffusion_setting=settings.lcm_diffusion_setting,
            )
        print(f"Latency : {elapsed:.2f} seconds")
        return images
