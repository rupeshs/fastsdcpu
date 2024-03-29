import numpy as np
from backend.annotators.control_interface import ControlInterface
from cv2 import Canny
from PIL import Image


class ResizeControl(ControlInterface):
    def get_control_image(
        self,
        input_image: Image,
    ) -> Image:
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(1024) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        return input_image.resize((W, H), resample=Image.LANCZOS)
