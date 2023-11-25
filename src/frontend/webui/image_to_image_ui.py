from typing import Any
import gradio as gr

from backend.models.lcmdiffusion_setting import LCMDiffusionSetting, DiffusionTask
from context import Context
from models.interface_types import InterfaceType
from app_settings import Settings
from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO
from frontend.utils import is_reshape_required
from app_settings import AppSettings
from constants import DEVICE
from frontend.utils import enable_openvino_controls

random_enabled = True

context = Context(InterfaceType.WEBUI)
previous_width = 0
previous_height = 0
previous_model_id = ""
previous_num_of_images = 0


def generate_image_to_image(
    prompt,
    init_image,
    strength,
    image_height,
    image_width,
    inference_steps,
    guidance_scale,
    num_images,
    seed,
    use_seed,
    use_openvino,
    use_safety_checker,
    tiny_auto_encoder_checkbox,
) -> Any:
    global previous_height, previous_width, previous_model_id, previous_num_of_images
    model_id = LCM_DEFAULT_MODEL
    if use_openvino:
        model_id = LCM_DEFAULT_MODEL_OPENVINO

    lcm_diffusion_settings = LCMDiffusionSetting(
        init_image=init_image,
        strength=strength,
        lcm_model_id=model_id,
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
        use_tiny_auto_encoder=tiny_auto_encoder_checkbox,
        diffusion_task=DiffusionTask.image_to_image.value,
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
            previous_num_of_images,
            num_images,
        )
    images = context.generate_text_to_image(
        settings,
        reshape,
        DEVICE,
    )
    previous_width = image_width
    previous_height = image_height
    previous_model_id = model_id
    previous_num_of_images = num_images

    return images


def get_image_to_image_ui(app_settings: AppSettings) -> None:
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
                input_image = gr.Image(label="Init image", type="pil")
                strength = gr.Slider(0, 1, value=0.8, step=0.1, label="Strength")
                num_inference_steps = gr.Slider(
                    1, 25, value=4, step=1, label="Inference Steps"
                )

                with gr.Accordion("Advanced options", open=False):
                    guidance_scale = gr.Slider(
                        1.0, 2.0, value=1.0, step=0.5, label="Guidance Scale"
                    )
                    image_height = gr.Slider(
                        256, 768, value=512, step=256, label="Image Height"
                    )
                    image_width = gr.Slider(
                        256, 768, value=512, step=256, label="Image Width"
                    )
                    num_images = gr.Slider(
                        1,
                        50,
                        value=1,
                        step=1,
                        label="Number of images to generate",
                    )

                    seed = gr.Slider(
                        value=123123,
                        minimum=0,
                        maximum=999999999,
                        label="Seed",
                        step=1,
                    )
                    seed_checkbox = gr.Checkbox(
                        label="Use random seed",
                        value=True,
                        interactive=True,
                    )

                    openvino_checkbox = gr.Checkbox(
                        label="Use OpenVINO",
                        value=False,
                        interactive=enable_openvino_controls(),
                    )

                    safety_checker_checkbox = gr.Checkbox(
                        label="Use Safety Checker",
                        value=True,
                        interactive=True,
                    )
                    tiny_auto_encoder_checkbox = gr.Checkbox(
                        label="Use tiny auto encoder for SD",
                        value=False,
                        interactive=True,
                    )

                    input_params = [
                        prompt,
                        input_image,
                        strength,
                        image_height,
                        image_width,
                        num_inference_steps,
                        guidance_scale,
                        num_images,
                        seed,
                        seed_checkbox,
                        openvino_checkbox,
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
        fn=generate_image_to_image,
        inputs=input_params,
        outputs=output,
    )
