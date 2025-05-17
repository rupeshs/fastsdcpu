from pprint import pprint
from time import perf_counter
from traceback import print_exc
from typing import Any

from app_settings import Settings
from backend.image_saver import ImageSaver
from backend.lcm_text_to_image import LCMTextToImage
from backend.models.lcmdiffusion_setting import DiffusionTask
from backend.utils import get_blank_image
from models.interface_types import InterfaceType


class Context:
    def __init__(
        self,
        interface_type: InterfaceType,
        device="cpu",
    ):
        self.interface_type = interface_type.value
        self.lcm_text_to_image = LCMTextToImage(device)
        self._latency = 0
        self._error = ""

    @property
    def latency(self):
        return self._latency

    @property
    def error(self):
        return self._error

    def generate_text_to_image(
        self,
        settings: Settings,
        reshape: bool = False,
        device: str = "cpu",
        save_config=True,
    ) -> Any:
        try:
            self._error = ""
            tick = perf_counter()
            from state import get_settings

            if (
                settings.lcm_diffusion_setting.diffusion_task
                == DiffusionTask.text_to_image.value
            ):
                settings.lcm_diffusion_setting.init_image = None

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
            self._latency = elapsed
            print(f"Latency : {elapsed:.2f} seconds")
            if settings.lcm_diffusion_setting.controlnet:
                if settings.lcm_diffusion_setting.controlnet.enabled:
                    images.append(
                        settings.lcm_diffusion_setting.controlnet._control_image
                    )

            if settings.lcm_diffusion_setting.use_safety_checker:
                print("Safety Checker is enabled")
                from state import get_safety_checker

                safety_checker = get_safety_checker()
                blank_image = get_blank_image(
                    settings.lcm_diffusion_setting.image_width,
                    settings.lcm_diffusion_setting.image_height,
                )
                for idx, image in enumerate(images):
                    if not safety_checker.is_safe(image):
                        images[idx] = blank_image
        except Exception as exception:
            print(f"Error in generating images: {exception}")
            self._error = str(exception)
            print_exc()
            return None
        return images

    def save_images(
        self,
        images: Any,
        settings: Settings,
    ) -> list[str]:
        saved_images = []
        if images and settings.generated_images.save_image:
            saved_images = ImageSaver.save_images(
                settings.generated_images.path,
                images=images,
                lcm_diffusion_setting=settings.lcm_diffusion_setting,
                format=settings.generated_images.format,
                jpeg_quality=settings.generated_images.save_image_quality,
            )
        return saved_images
