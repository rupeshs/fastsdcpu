from pydantic import BaseModel
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting, LCMLora
from backend.models.gen_images import GeneratedImages


class Settings(BaseModel):
    lcm_diffusion_setting: LCMDiffusionSetting = LCMDiffusionSetting(lcm_lora=LCMLora())
    generated_images: GeneratedImages = GeneratedImages()
