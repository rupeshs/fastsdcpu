def is_reshape_required(
    prev_width: int,
    cur_width: int,
    prev_height: int,
    cur_height: int,
    prev_model: int,
    cur_model: int,
) -> bool:
    print(f"width - {prev_width} {cur_width}")
    print(f"height - {prev_height} {cur_height}")
    print(f"model - {prev_model} {cur_model}")
    reshape_required = False
    if prev_width != cur_width or prev_height != cur_height or prev_model != cur_model:
        print("Reshape and compile")
        reshape_required = True

    return reshape_required
