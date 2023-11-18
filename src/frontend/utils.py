from constants import DEVICE
from typing import List
import platform
from backend.device import is_openvino_device


def is_reshape_required(
    prev_width: int,
    cur_width: int,
    prev_height: int,
    cur_height: int,
    prev_model: int,
    cur_model: int,
    prev_num_of_images: int,
    cur_num_of_images: int,
) -> bool:
    reshape_required = False
    if (
        prev_width != cur_width
        or prev_height != cur_height
        or prev_model != cur_model
        or prev_num_of_images != cur_num_of_images
    ):
        print("Reshape and compile")
        reshape_required = True

    return reshape_required


def enable_openvino_controls() -> bool:
    return is_openvino_device() and platform.system().lower() != "darwin"


def get_valid_model_id(
    models: List,
    model_id: str,
) -> str:
    if len(models) == 0:
        print("Error: model configuration file is empty,please add some models.")
        return ""
    if model_id == "":
        return models[0]

    if model_id in models:
        return model_id
    else:
        print(
            f"Error:{model_id} Model not found in configuration file,so using first model : {models[0]}"
        )
        return models[0]
