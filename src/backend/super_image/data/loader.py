import numpy as np
import cv2
from PIL import Image

import torch
from torch import Tensor


class ImageLoader:
    @staticmethod
    def load_image(image: Image):
        lr = np.array(image.convert('RGB'))
        lr = lr[::].astype(np.float32).transpose([2, 0, 1]) / 255.0
        return torch.as_tensor(np.array([lr]))

    @staticmethod
    def _process_image_to_save(pred: Tensor):
        pred = pred.data.cpu().numpy()
        pred = pred[0].transpose((1, 2, 0)) * 255.0
        # pred = pred[scale:-scale, scale:-scale, :]
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        return pred

    @staticmethod
    def save_image(pred: Tensor, output_file: str):
        pred = ImageLoader._process_image_to_save(pred)
        cv2.imwrite(output_file, pred)

    @staticmethod
    def save_compare(input: Tensor, pred: Tensor, output_file: str):
        pred = ImageLoader._process_image_to_save(pred)
        input = ImageLoader._process_image_to_save(input)
        input_resize = cv2.resize(input, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_CUBIC)
        hstack = np.hstack((input_resize, pred))
        cv2.imwrite(output_file, hstack)
