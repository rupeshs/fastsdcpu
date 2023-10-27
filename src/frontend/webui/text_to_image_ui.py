from typing import Any
import gradio as gr

from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
from pprint import pprint
from context import Context
from models.interface_types import InterfaceType
from app_settings import Settings
from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO
from frontend.utils import is_reshape_required
from os import environ

random_enabled = True

context = Context(InterfaceType.WEBUI)
previous_width = 0
previous_height = 0
previous_model_id = ""


def generate_text_to_image(
    prompt,
    image_height,
    image_width,
    inference_steps,
    guidance_scale,
    num_images,
    seed,
    use_openvino,
    use_safety_checker,
) -> Any:
    global previous_height, previous_width, previous_model_id
    model_id = LCM_DEFAULT_MODEL
    if use_openvino:
        model_id = LCM_DEFAULT_MODEL_OPENVINO

    use_seed = True if seed != -1 else False

    lcm_diffusion_settings = LCMDiffusionSetting(
        model_id=model_id,
        prompt=prompt,
        image_height=image_height,
        image_width=image_width,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        number_of_images=num_images,
        seed=seed,
        use_openvino=use_openvino,
        use_safety_checker=use_safety_checker,
        use_seed=use_seed,
    )
    settings = Settings(
        lcm_diffusion_setting=lcm_diffusion_settings,
    )
    reshape = False
    if use_openvino:
        reshape = is_reshape_required(
            previous_width,
            image_width,
            previous_height,
            image_height,
            previous_model_id,
            model_id,
        )

    pprint(lcm_diffusion_settings.model_dump())

    images = context.generate_text_to_image(
        settings,
        reshape,
        environ.get("DEVICE", "cpu"),
    )
    previous_width = image_width
    previous_height = image_height
    previous_model_id = model_id

    return images


def get_text_to_image_ui() -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():

                def random_seed():
                    global random_enabled
                    random_enabled = not random_enabled
                    seed_val = -1
                    if not random_enabled:
                        seed_val = 42
                    return gr.Number.update(
                        interactive=not random_enabled, value=seed_val
                    )

                prompt = gr.Textbox(
                    label="Describe the image you'd like to see",
                    lines=3,
                    placeholder="A fantasy landscape",
                )
                generate_btn = gr.Button("Generate", elem_id="generate_button")

                with gr.Accordion("Advanced options", open=False):
                    image_height = gr.Slider(
                        256, 768, value=512, step=256, label="Image Height"
                    )
                    image_width = gr.Slider(
                        256, 768, value=512, step=256, label="Image Width"
                    )
                    num_inference_steps = gr.Slider(
                        1, 25, value=4, step=1, label="Inference Steps"
                    )

                    guidance_scale = gr.Slider(
                        1.0, 30.0, value=8, step=0.5, label="Guidance Scale"
                    )
                    num_images = gr.Slider(
                        1,
                        50,
                        value=1,
                        step=1,
                        label="Number of images to generate",
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        interactive=False,
                    )
                    seed_checkbox = gr.Checkbox(
                        label="Use random seed",
                        value=True,
                        interactive=True,
                    )

                    openvino_checkbox = gr.Checkbox(
                        label="Use OpenVINO",
                        value=False,
                        interactive=True,
                    )

                    safety_checker_checkbox = gr.Checkbox(
                        label="Use Safety Checker",
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
                        openvino_checkbox,
                        safety_checker_checkbox,
                    ]

            with gr.Column():
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                )

    seed_checkbox.change(fn=random_seed, outputs=seed)
    generate_btn.click(
        fn=generate_text_to_image,
        inputs=input_params,
        outputs=output,
    )
