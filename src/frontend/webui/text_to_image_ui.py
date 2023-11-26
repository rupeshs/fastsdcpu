import gradio as gr
from typing import Any
from backend.models.lcmdiffusion_setting import DiffusionTask
from context import Context
from models.interface_types import InterfaceType
from constants import DEVICE
from state import get_settings
from frontend.utils import is_reshape_required
from concurrent.futures import ThreadPoolExecutor

app_settings = get_settings()
context = Context(InterfaceType.WEBUI)
previous_width = 0
previous_height = 0
previous_model_id = ""
previous_num_of_images = 0


def generate_text_to_image(
    prompt,
    image_height,
    image_width,
    inference_steps,
    guidance_scale,
    num_images,
    seed,
    use_seed,
    use_safety_checker,
    tiny_auto_encoder_checkbox,
) -> Any:
    global previous_height, previous_width, previous_model_id, previous_num_of_images, app_settings

    app_settings.settings.lcm_diffusion_setting.prompt = prompt
    app_settings.settings.lcm_diffusion_setting.image_height = image_height
    app_settings.settings.lcm_diffusion_setting.image_width = image_width
    app_settings.settings.lcm_diffusion_setting.inference_steps = inference_steps
    app_settings.settings.lcm_diffusion_setting.guidance_scale = guidance_scale
    app_settings.settings.lcm_diffusion_setting.number_of_images = num_images
    app_settings.settings.lcm_diffusion_setting.seed = seed
    app_settings.settings.lcm_diffusion_setting.use_seed = use_seed
    app_settings.settings.lcm_diffusion_setting.use_safety_checker = use_safety_checker
    app_settings.settings.lcm_diffusion_setting.use_tiny_auto_encoder = (
        tiny_auto_encoder_checkbox
    )
    app_settings.settings.lcm_diffusion_setting.diffusion_task = (
        DiffusionTask.text_to_image.value
    )
    model_id = app_settings.settings.lcm_diffusion_setting.openvino_lcm_model_id
    reshape = False
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

    executor = ThreadPoolExecutor()
    future = executor.submit(
        context.generate_text_to_image,
        app_settings.settings,
        reshape,
        DEVICE,
    )
    images = future.result()
    # images = context.generate_text_to_image(
    #     app_settings.settings,
    #     reshape,
    #     DEVICE,
    # )

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
                        label="Describe the image you'd like to see",
                        lines=3,
                        placeholder="A fantasy landscape",
                    )

                    generate_btn = gr.Button(
                        "Generate",
                        elem_id="generate_button",
                        scale=0,
                    )
                num_inference_steps = gr.Slider(
                    1, 25, value=4, step=1, label="Inference Steps"
                )
                image_height = gr.Slider(
                    256, 1024, value=512, step=256, label="Image Height"
                )
                image_width = gr.Slider(
                    256, 1024, value=512, step=256, label="Image Width"
                )
                num_images = gr.Slider(
                    1,
                    50,
                    value=1,
                    step=1,
                    label="Number of images to generate",
                )
                with gr.Accordion("Advanced options", open=False):
                    guidance_scale = gr.Slider(
                        1.0, 2.0, value=1.0, step=0.5, label="Guidance Scale"
                    )

                    seed = gr.Slider(
                        value=123123,
                        minimum=0,
                        maximum=999999999,
                        label="Seed",
                        step=1,
                    )
                    seed_checkbox = gr.Checkbox(
                        label="Use seed",
                        value=False,
                        interactive=True,
                    )

                    safety_checker_checkbox = gr.Checkbox(
                        label="Use Safety Checker",
                        value=False,
                        interactive=True,
                    )
                    tiny_auto_encoder_checkbox = gr.Checkbox(
                        label="Use tiny auto encoder for SD",
                        value=False,
                        interactive=True,
                    )

                    input_params = [
                        prompt,
                        image_height,
                        image_width,
                        num_inference_steps,
                        guidance_scale,
                        num_images,
                        seed,
                        seed_checkbox,
                        safety_checker_checkbox,
                        tiny_auto_encoder_checkbox,
                    ]

            with gr.Column():
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                )

    # seed_checkbox.change(fn=random_seed, outputs=seed)
    generate_btn.click(
        fn=generate_text_to_image,
        inputs=input_params,
        outputs=output,
    )
