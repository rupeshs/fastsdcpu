from enum import Enum


class UpscaleMode(str, Enum):
    """Diffusion task types"""

    normal = "normal"
    sd_upscale = "sd_upscale"
