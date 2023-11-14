import platform
from constants import DEVICE
import torch


def get_device_name() -> str:
    if DEVICE == "cuda":
        default_gpu_index = torch.cuda.current_device()
        return torch.cuda.get_device_name(default_gpu_index)
    else:
        return platform.processor()
