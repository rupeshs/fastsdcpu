import platform
from constants import DEVICE
import torch
import openvino as ov

core = ov.Core()


def is_openvino_device() -> bool:
    if DEVICE.lower() == "cpu" or DEVICE.lower()[0] == "g" or DEVICE.lower()[0] == "n":
        return True
    else:
        return False


def get_device_name() -> str:
    if DEVICE == "cuda" or DEVICE == "mps":
        default_gpu_index = torch.cuda.current_device()
        return torch.cuda.get_device_name(default_gpu_index)
    elif platform.system().lower() == "darwin":
        return platform.processor()
    elif is_openvino_device():
        return core.get_property(DEVICE.upper(), "FULL_DEVICE_NAME")
