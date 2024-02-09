from pydantic import BaseModel
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting, LCMLora
from backend.models.gen_images import GeneratedImages
from paths import FastStableDiffusionPaths


class Settings(BaseModel):
    results_path: str = FastStableDiffusionPaths().get_results_path()
    lcm_diffusion_setting: LCMDiffusionSetting = LCMDiffusionSetting(lcm_lora=LCMLora())
    generated_images: GeneratedImages = GeneratedImages()
