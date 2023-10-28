from pydantic import BaseModel, Field
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
from paths import FastStableDiffusionPaths


class Settings(BaseModel):
    results_path: str = FastStableDiffusionPaths().get_results_path()
    lcm_setting: LCMDiffusionSetting = LCMDiffusionSetting().dict()
