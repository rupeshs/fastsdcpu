from backend.annotators.control_interface import ControlInterface
from controlnet_aux import ContentShuffleDetector
from PIL import Image


class ShuffleControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        shuffle_processor = ContentShuffleDetector()
        image = shuffle_processor(image)
        return image
