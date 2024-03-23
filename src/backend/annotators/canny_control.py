import numpy as np
from cv2 import Canny
from PIL import Image

from backend.annotators.control_interface import ControlInterface


class CannyControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        low_threshold = 100
        high_threshold = 200
        image = np.array(image)
        image = Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)
