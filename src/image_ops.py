from PIL import Image


def resize_pil_image(
    pil_image: Image,
    image_width,
    image_height,
):
    return pil_image.convert("RGB").resize(
        (
            image_width,
            image_height,
        ),
        Image.Resampling.LANCZOS,
    )
