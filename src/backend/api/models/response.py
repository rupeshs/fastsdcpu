from typing import List

from pydantic import BaseModel


class StableDiffusionResponse(BaseModel):
    """
    Stable diffusion response model

    Attributes:
        images (List[str]): List of JPEG image as base64 encoded
        latency (float): Latency in seconds
        error (str): Error message if any
    """

    images: List[str]
    latency: float
    error: str = ""
