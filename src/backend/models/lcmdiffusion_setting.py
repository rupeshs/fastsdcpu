from typing import Optional

from pydantic import BaseModel
from constants import LCM_DEFAULT_MODEL


class LCMLora(BaseModel):
    base_model_id: str = ""
    lcm_lora_id: str = ""


class LCMDiffusionSetting(BaseModel):
    lcm_model_id: str = LCM_DEFAULT_MODEL
    prompt: str = ""
    negative_prompt: str = ""
    image_height: Optional[int] = 512
    image_width: Optional[int] = 512
    inference_steps: Optional[int] = 4
    guidance_scale: Optional[float] = 1
    number_of_images: Optional[int] = 1
    seed: Optional[int] = -1
    use_openvino: bool = False
    use_seed: bool = False
    use_offline_model: bool = False
    use_safety_checker: bool = False
    use_tiny_auto_encoder: bool = False
    use_lcm_lora: bool = False
    lcm_lora: Optional[LCMLora] = LCMLora()
