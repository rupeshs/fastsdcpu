from typing import Any
import gradio as gr
from models.interface_types import InterfaceType
from constants import DEVICE
from state import get_settings, get_context


app_settings = get_settings()

context = get_context(InterfaceType.WEBUI)
previous_width = 0
previous_height = 0
previous_model_id = ""
previous_num_of_images = 0


def generate_image_to_image(
    init_image,
) -> Any:

    return None


def get_upscaler_ui() -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Image", type="pil")
                with gr.Row():
                    generate_btn = gr.Button(
                        "Upscale",
                        elem_id="generate_button",
                        scale=0,
                    )

                input_params = [
                    input_image,
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
        fn=generate_image_to_image,
        inputs=input_params,
        outputs=output,
    )
