import numpy as np
from backend.annotators.control_interface import ControlInterface
from controlnet_aux import LineartDetector
from PIL import Image


class LineArtControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        control_image = processor(image)
        return control_image
