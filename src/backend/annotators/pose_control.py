from backend.annotators.control_interface import ControlInterface
from controlnet_aux import OpenposeDetector
from PIL import Image


class PoseControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        image = openpose(image)
        return image
