from pydantic import BaseModel
from enum import Enum
from paths import FastStableDiffusionPaths


class ImageFormat(str, Enum):
    """Image format"""

    JPEG = "JPEG"
    PNG = "PNG"


class GeneratedImages(BaseModel):
    path: str = FastStableDiffusionPaths.get_results_path()
    format: str = ImageFormat.PNG.value
    save_image: bool = True
