from io import BytesIO
from base64 import b64encode, b64decode
from PIL import Image


def pil_image_to_base64_str(
    image: Image,
    format: str = "JPEG",
) -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    img_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64


def base64_image_to_pil(base64_str) -> Image:
    image_data = b64decode(base64_str)
    image_buffer = BytesIO(image_data)
    image = Image.open(image_buffer)
    return image
