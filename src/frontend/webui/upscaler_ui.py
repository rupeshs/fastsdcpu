from typing import Any
import gradio as gr
from models.interface_types import InterfaceType
from state import get_settings, get_context
from backend.upscale.upscaler import upscale_image
from backend.models.upscale import UpscaleMode
from paths import FastStableDiffusionPaths, join_paths
from time import time

app_settings = get_settings()


previous_width = 0
previous_height = 0
previous_model_id = ""
previous_num_of_images = 0


def create_upscaled_image(
    source_image,
    upscale_mode,
) -> Any:
    context = get_context(InterfaceType.WEBUI)
    scale_factor = 2
    extension = "png"
    if upscale_mode == "SD":
        mode = UpscaleMode.sd_upscale.value
    else:
        mode = UpscaleMode.normal.value

    upscaled_filepath = join_paths(
        FastStableDiffusionPaths.get_results_path(),
        f"fastsdcpu_{int(scale_factor)}x_upscale_{int(time())}.{extension}",
    )
    image = upscale_image(
        context=context,
        src_image=source_image,
        dst_image_path=upscaled_filepath,
        upscale_mode=mode,
    )
    return image


def get_upscaler_ui() -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Image", type="pil")
                with gr.Row():
                    upscale_mode = gr.Radio(
                        ["EDSR", "SD"],
                        label="Upscale Mode (2x)",
                        info="Select upscale method, SD Upscale is experimental",
                        value="EDSR",
                    )

                    generate_btn = gr.Button(
                        "Upscale",
                        elem_id="generate_button",
                        scale=0,
                    )

                input_params = [
                    input_image,
                    upscale_mode,
                ]

            with gr.Column():
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    height=512,
                )

    generate_btn.click(
        fn=create_upscaled_image,
        inputs=input_params,
        outputs=output,
    )
