from backend.super_image import EdsrModel, ImageLoader
from PIL import Image

UPSCALE_MODEL = "eugenesiow/edsr-base"


def upscale_image(
    src_image: Image,
    dst_image_path: str,
    scale_factor: int = 2,
):
    model = EdsrModel.from_pretrained(UPSCALE_MODEL, scale=scale_factor)
    inputs = ImageLoader.load_image(src_image)
    preds = model(inputs)
    ImageLoader.save_image(preds, dst_image_path)
