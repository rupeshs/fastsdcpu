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
