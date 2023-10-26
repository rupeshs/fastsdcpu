from typing import Any
import gradio as gr
from backend.lcm_text_to_image import LCMTextToImage
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO
from pprint import pprint

random_enabled = True
lcm_text_to_image = LCMTextToImage()


def generate_text_to_image(
    prompt,
    image_height,
    image_width,
    inference_steps,
    guidance_scale,
    seed,
    use_openvino,
    use_safety_checker,
) -> Any:
    lcm_diffusion_settings = LCMDiffusionSetting(
        prompt=prompt,
        image_height=image_height,
        image_width=image_width,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        # number_of_images=num_images,
        seed=seed,
        use_openvino=use_openvino,
        use_safety_checker=use_safety_checker,
    )
    pprint(lcm_diffusion_settings.model_dump())
    if use_openvino:
        lcm_text_to_image.init(
            LCM_DEFAULT_MODEL_OPENVINO,
            use_openvino,
        )
    else:
        lcm_text_to_image.init(
            LCM_DEFAULT_MODEL,
            use_openvino,
        )

    images = lcm_text_to_image.generate(lcm_diffusion_settings)
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
                    # num_images = gr.Slider(
                    #     1,
                    #     50,
                    #     value=1,
                    #     step=1,
                    #     label="Number of images to generate",
                    # )

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
                        # num_images,
                        seed,
                        openvino_checkbox,
                        safety_checker_checkbox,
                    ]

            with gr.Column():
                generate_btn = gr.Button("Generate", elem_id="generate_button")
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
