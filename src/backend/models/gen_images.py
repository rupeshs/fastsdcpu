from pydantic import BaseModel
from enum import Enum
from paths import FastStableDiffusionPaths


class ImageFormat(str, Enum):
    """Image format"""

    JPEG = "jpeg"
    PNG = "png"


class GeneratedImages(BaseModel):
    path: str = FastStableDiffusionPaths.get_results_path()
    format: str = ImageFormat.PNG.value.upper()
    save_image: bool = True
    save_image_quality: int = 90
