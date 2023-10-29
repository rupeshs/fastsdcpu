from constants import DEVICE


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
    print(f"width - {prev_width} {cur_width}")
    print(f"height - {prev_height} {cur_height}")
    print(f"model - {prev_model} {cur_model}")
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
    return DEVICE == "cpu"
