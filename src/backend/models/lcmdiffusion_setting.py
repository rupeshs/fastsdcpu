from typing import Optional, Any

from pydantic import BaseModel
from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO


class LCMLora(BaseModel):
    base_model_id: str = ""
    lcm_lora_id: str = ""


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
    strength: Optional[float] = 0.75
    image_height: Optional[int] = 512
    image_width: Optional[int] = 512
    inference_steps: Optional[int] = 4
    guidance_scale: Optional[float] = 1
    number_of_images: Optional[int] = 1
    seed: Optional[int] = -1
    use_seed: bool = False
    use_safety_checker: bool = False
