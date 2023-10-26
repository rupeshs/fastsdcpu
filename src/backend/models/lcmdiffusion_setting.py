from typing import Optional

from pydantic import BaseModel
from constants import LCM_DEFAULT_MODEL


class LCMDiffusionSetting(BaseModel):
    model_id: str = LCM_DEFAULT_MODEL
    prompt: str = ""
    image_height: Optional[int] = 512
    image_width: Optional[int] = 512
    inference_steps: Optional[int] = 4
    lcm_origin_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 8
    number_of_images: Optional[int] = 1
    seed: Optional[int] = -1
    use_openvino: bool = False
    use_seed: bool = False
    use_offline_model: bool = False
    use_safety_checker: bool = True
