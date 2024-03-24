import numpy as np
from backend.annotators.control_interface import ControlInterface
from PIL import Image
from transformers import pipeline


class DepthControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        depth_estimator = pipeline("depth-estimation")
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image
