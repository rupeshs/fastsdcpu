import gradio as gr
from typing import Any
from backend.models.lcmdiffusion_setting import DiffusionTask
from models.interface_types import InterfaceType
from constants import DEVICE
from state import get_settings, get_context
from frontend.utils import is_reshape_required
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

app_settings = get_settings()

previous_width = 0
previous_height = 0
previous_model_id = ""
previous_num_of_images = 0


def generate_text_to_image(
    prompt,
    neg_prompt,
) -> Any:
    context = get_context(InterfaceType.WEBUI)
    global previous_height, previous_width, previous_model_id, previous_num_of_images, app_settings
    app_settings.settings.lcm_diffusion_setting.prompt = prompt
    app_settings.settings.lcm_diffusion_setting.negative_prompt = neg_prompt
    app_settings.settings.lcm_diffusion_setting.diffusion_task = (
        DiffusionTask.text_to_image.value
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


def get_text_to_image_ui() -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Textbox(
                        show_label=False,
                        lines=3,
                        placeholder="A fantasy landscape",
                        container=False,
                    )

                    generate_btn = gr.Button(
                        "Generate",
                        elem_id="generate_button",
                        scale=0,
                    )
                negative_prompt = gr.Textbox(
                    label="Negative prompt (Works in LCM-LoRA mode, set guidance > 1.0) :",
                    lines=1,
                    placeholder="",
                )

                input_params = [prompt, negative_prompt]

            with gr.Column():
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    height=512,
                )
    generate_btn.click(
        fn=generate_text_to_image,
        inputs=input_params,
        outputs=output,
    )
