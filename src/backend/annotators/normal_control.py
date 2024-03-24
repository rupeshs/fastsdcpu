from backend.annotators.control_interface import ControlInterface
from controlnet_aux import NormalBaeDetector
from PIL import Image


class NormalControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        control_image = processor(image)
        return control_image
