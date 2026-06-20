from typing import Any
import gradio as gr
from backend.models.lcmdiffusion_setting import DiffusionTask
from models.interface_types import InterfaceType
from frontend.utils import is_reshape_required
from constants import DEVICE
from state import get_settings, get_context, get_edit_image_prompts
from concurrent.futures import ThreadPoolExecutor
from frontend.webui.errors import show_error

app_settings = get_settings()
image_edit_prompts = get_edit_image_prompts()

previous_width = 0
previous_height = 0
previous_model_id = ""
previous_num_of_images = 0


def edit_image(
    prompt,
    init_image,
) -> Any:
    context = get_context(InterfaceType.WEBUI)
    global \
        previous_height, \
        previous_width, \
        previous_model_id, \
        previous_num_of_images, \
        app_settings

    app_settings.settings.lcm_diffusion_setting.prompt = prompt
    app_settings.settings.lcm_diffusion_setting.negative_prompt = ""
    app_settings.settings.lcm_diffusion_setting.init_image = init_image

    app_settings.settings.lcm_diffusion_setting.diffusion_task = (
        DiffusionTask.edit_image.value
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
        if images:
            context.save_images(
                images,
                app_settings.settings,
            )
        else:
            show_error(context.error)

    previous_width = image_width
    previous_height = image_height
    previous_model_id = model_id
    previous_num_of_images = num_images
    return images


display_to_key = {
    prompt_config["display_name"]: prompt_key
    for prompt_key, prompt_config in image_edit_prompts.items()
}


def update_prompt(selected_prompt):
    prompt = image_edit_prompts.get(display_to_key[selected_prompt]).get("prompt", "")
    return prompt


def get_edit_prompts_presets():
    display_to_key = [value["display_name"] for _, value in image_edit_prompts.items()]
    return display_to_key


def get_edit_image_ui() -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Init image", type="pil")
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
                default_prompt = gr.Dropdown(
                    label="Select prompt ",
                    choices=get_edit_prompts_presets(),
                    value="None",
                    interactive=True,
                )

                input_params = [
                    prompt,
                    input_image,
                ]
                default_prompt.change(
                    fn=update_prompt,
                    inputs=default_prompt,
                    outputs=prompt,
                )

            with gr.Column():
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    height=512,
                )

    generate_btn.click(
        fn=edit_image,
        inputs=input_params,
        outputs=output,
    )
