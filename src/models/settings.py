from pydantic import BaseModel
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
from paths import FastStableDiffusionPaths


class Settings(BaseModel):
    results_path: str = FastStableDiffusionPaths().get_results_path()
    lcm_diffusion_setting: LCMDiffusionSetting = LCMDiffusionSetting()
