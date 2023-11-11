from typing import List
from constants import LCM_DEFAULT_MODEL


def get_available_models() -> List:
    models = [
        LCM_DEFAULT_MODEL,
        "latent-consistency/lcm-sdxl",
        "latent-consistency/lcm-ssd-1b",
    ]
    return models
