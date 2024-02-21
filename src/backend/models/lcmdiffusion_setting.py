from typing import Optional, Any
from enum import Enum
from pydantic import BaseModel
from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO


class LCMLora(BaseModel):
    base_model_id: str = "Lykon/dreamshaper-8"
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"


class DiffusionTask(str, Enum):
    """Diffusion task types"""

    text_to_image = "text_to_image"
    image_to_image = "image_to_image"


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
    lora_path: Any = None
