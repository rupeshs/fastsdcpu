from backend.upscale.aura_sr import AuraSR
from PIL import Image


def upscale_aura_sr(image_path: str):
    aura_sr = AuraSR.from_pretrained("fal-ai/AuraSR", device="cpu")
    image_in = Image.open(image_path).resize((256, 256))
    return aura_sr.upscale_4x(image_in)
