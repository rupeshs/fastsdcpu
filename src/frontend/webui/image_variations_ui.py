from typing import Any
import gradio as gr
from backend.models.lcmdiffusion_setting import DiffusionTask
from context import Context
from models.interface_types import InterfaceType
from frontend.utils import is_reshape_required
from constants import DEVICE
from state import get_settings, get_context
from concurrent.futures import ThreadPoolExecutor

app_settings = get_settings()


previous_width = 0
previous_height = 0
previous_model_id = ""
previous_num_of_images = 0


def generate_image_variations(
    init_image,
    variation_strength,
) -> Any:
    context = get_context(InterfaceType.WEBUI)
    global previous_height, previous_width, previous_model_id, previous_num_of_images, app_settings

    app_settings.settings.lcm_diffusion_setting.init_image = init_image
    app_settings.settings.lcm_diffusion_setting.strength = variation_strength
    app_settings.settings.lcm_diffusion_setting.prompt = ""
    app_settings.settings.lcm_diffusion_setting.negative_prompt = ""

    app_settings.settings.lcm_diffusion_setting.diffusion_task = (
        DiffusionTask.image_to_image.value
    )
    model_id = app_settings.settings.lcm_diffusion_setting.openvino_lcm_model_id
    reshape = False
    image_width = app_settings.settings.lcm_diffusion_setting.image_width
    image_height = app_settings.settings.lcm_diffusion_setting.image_height
    num_images = app_settings.settings.lcm_diffusion_setting.number_of_images
    if app_settings.settings.lcm_diffusion_setting.use_openvino:
        reshape = is_reshape_required(
            previous_width,
            image_width,
            previous_height,
            image_height,
            previous_model_id,
            model_id,
            previous_num_of_images,
            num_images,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            context.generate_text_to_image,
            app_settings.settings,
            reshape,
            DEVICE,
        )
        images = future.result()

    previous_width = image_width
    previous_height = image_height
    previous_model_id = model_id
    previous_num_of_images = num_images
    return images


def get_image_variations_ui() -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Init image", type="pil")
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate",
                        elem_id="generate_button",
                        scale=0,
                    )

                variation_strength = gr.Slider(
                    0.1,
                    1,
                    value=0.4,
                    step=0.01,
                    label="Variations Strength",
                )

                input_params = [
                    input_image,
                    variation_strength,
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
        fn=generate_image_variations,
        inputs=input_params,
        outputs=output,
    )
