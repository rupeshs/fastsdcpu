from backend.annotators.control_interface import ControlInterface
from controlnet_aux import MLSDdetector
from PIL import Image


class MlsdControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        image = mlsd(image)
        return image
