from backend.models.lcmdiffusion_setting import DiffusionTask
from backend.models.upscale import UpscaleMode
from backend.upscale.edsr_upscale_onnx import upscale_edsr_2x
from backend.upscale.tiled_upscale import generate_upscaled_image
from context import Context
from PIL import Image
from state import get_settings


config = get_settings()


def upscale_image(
    context: Context,
    src_image_path: str,
    dst_image_path: str,
    scale_factor: int = 2,
    upscale_mode: UpscaleMode = UpscaleMode.normal.value,
):
    if upscale_mode == UpscaleMode.normal.value:

        upscaled_img = upscale_edsr_2x(src_image_path)
        upscaled_img.save(dst_image_path)
        print(f"Upscaled image saved {dst_image_path}")
    else:
        config.settings.lcm_diffusion_setting.strength = (
            0.3 if config.settings.lcm_diffusion_setting.use_openvino else 0.1
        )
        config.settings.lcm_diffusion_setting.diffusion_task = (
            DiffusionTask.image_to_image.value
        )

        generate_upscaled_image(
            config.settings,
            src_image_path,
            config.settings.lcm_diffusion_setting.strength,
            upscale_settings=None,
            context=context,
            tile_overlap=(
                32 if config.settings.lcm_diffusion_setting.use_openvino else 16
            ),
            output_path=dst_image_path,
            image_format=config.settings.generated_images.format,
        )
        print(f"Upscaled image saved {dst_image_path}")

    return [Image.open(dst_image_path)]
