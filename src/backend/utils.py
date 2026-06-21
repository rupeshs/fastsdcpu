from PIL import Image


def get_blank_image(
    width: int,
    height: int,
) -> Image.Image:
    """
    Create a blank image with the specified width and height.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        Image.Image: A blank image with the specified dimensions.
    """
    return Image.new("RGB", (width, height), (0, 0, 0))


def get_image_edit_dimensions(img, max_size=1024) -> tuple[int, int]:
    """Update width/height sliders based on uploaded image aspect ratio"""

    img_width, img_height = img.size

    aspect_ratio = img_width / img_height

    print(
        f"Original image size: {img_width}x{img_height}, aspect ratio: {aspect_ratio:.2f}"
    )

    if aspect_ratio >= 1:  # Landscape or square
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:  # Portrait
        new_height = max_size
        new_width = int(max_size * aspect_ratio)

    # Round to nearest multiple of 8
    new_width = round(new_width / 8) * 8
    new_height = round(new_height / 8) * 8

    # Ensure within valid range (minimum 256, maximum 1024)
    new_width = max(256, min(max_size, new_width))
    new_height = max(256, min(max_size, new_height))

    print(
        f"Adjusted aspect ratio: {new_width}/{new_height} = {new_width / new_height:.2f}"
    )

    return new_width, new_height
