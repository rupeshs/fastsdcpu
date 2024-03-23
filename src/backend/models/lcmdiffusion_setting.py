from enum import Enum
from PIL import Image
from typing import Any, Optional, Union

from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO
from paths import FastStableDiffusionPaths
from pydantic import BaseModel


class LCMLora(BaseModel):
    base_model_id: str = "Lykon/dreamshaper-8"
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"


class DiffusionTask(str, Enum):
    """Diffusion task types"""

    text_to_image = "text_to_image"
    image_to_image = "image_to_image"


class Lora(BaseModel):
    models_dir: str = FastStableDiffusionPaths.get_lora_models_path()
    path: Optional[Any] = None
    weight: Optional[float] = 0.5
    fuse: bool = True
    enabled: bool = False


class ControlNetSetting(BaseModel):
    adapter_path: Optional[str] = None  # ControlNet adapter path
    conditioning_scale: float = 0.5
    enabled: bool = False
    _control_image: Image = None  # Control image, PIL image


class LCMDiffusionSetting(BaseModel):
    lcm_model_id: str = LCM_DEFAULT_MODEL
    openvino_lcm_model_id: str = LCM_DEFAULT_MODEL_OPENVINO
    use_offline_model: bool = False
    use_lcm_lora: bool = False
    lcm_lora: Optional[LCMLora] = LCMLora()
    use_tiny_auto_encoder: bool = False
    use_openvino: bool = False
    prompt: str = ""
    negative_prompt: str = ""
    init_image: Any = None
    strength: Optional[float] = 0.6
    image_height: Optional[int] = 512
    image_width: Optional[int] = 512
    inference_steps: Optional[int] = 1
    guidance_scale: Optional[float] = 1
    number_of_images: Optional[int] = 1
    seed: Optional[int] = 123123
    use_seed: bool = False
    use_safety_checker: bool = False
    diffusion_task: str = DiffusionTask.text_to_image.value
    lora: Optional[Lora] = Lora()
    controlnet: Optional[Union[ControlNetSetting, list[ControlNetSetting]]] = None
    dirs: dict = {
        "controlnet": FastStableDiffusionPaths.get_controlnet_models_path(),
        "lora": FastStableDiffusionPaths.get_lora_models_path(),
    }
    rebuild_pipeline: bool = False
