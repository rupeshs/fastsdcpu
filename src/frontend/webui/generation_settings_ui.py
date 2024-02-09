import gradio as gr
from state import get_settings
from backend.models.gen_images import ImageFormat

app_settings = get_settings()


def on_change_inference_steps(steps):
    app_settings.settings.lcm_diffusion_setting.inference_steps = steps


def on_change_image_width(img_width):
    app_settings.settings.lcm_diffusion_setting.image_width = img_width


def on_change_image_height(img_height):
    app_settings.settings.lcm_diffusion_setting.image_height = img_height


def on_change_num_images(num_images):
    app_settings.settings.lcm_diffusion_setting.number_of_images = num_images


def on_change_guidance_scale(guidance_scale):
    app_settings.settings.lcm_diffusion_setting.guidance_scale = guidance_scale


def on_change_seed_value(seed):
    app_settings.settings.lcm_diffusion_setting.seed = seed


def on_change_seed_checkbox(seed_checkbox):
    app_settings.settings.lcm_diffusion_setting.use_seed = seed_checkbox


def on_change_safety_checker_checkbox(safety_checker_checkbox):
    app_settings.settings.lcm_diffusion_setting.use_safety_checker = (
        safety_checker_checkbox
    )


def on_change_tiny_auto_encoder_checkbox(tiny_auto_encoder_checkbox):
    app_settings.settings.lcm_diffusion_setting.use_tiny_auto_encoder = (
        tiny_auto_encoder_checkbox
    )


def on_offline_checkbox(offline_checkbox):
    app_settings.settings.lcm_diffusion_setting.use_offline_model = offline_checkbox


def on_change_image_format(image_format):
    if image_format == "PNG":
        app_settings.settings.generated_images.format = ImageFormat.PNG.value.upper()
    else:
        app_settings.settings.generated_images.format = ImageFormat.JPEG.value.upper()

    app_settings.save()


def get_generation_settings_ui() -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                num_inference_steps = gr.Slider(
                    1,
                    25,
                    value=app_settings.settings.lcm_diffusion_setting.inference_steps,
                    step=1,
                    label="Inference Steps",
                    interactive=True,
                )

                image_height = gr.Slider(
                    256,
                    1024,
                    value=app_settings.settings.lcm_diffusion_setting.image_height,
                    step=256,
                    label="Image Height",
                    interactive=True,
                )
                image_width = gr.Slider(
                    256,
                    1024,
                    value=app_settings.settings.lcm_diffusion_setting.image_width,
                    step=256,
                    label="Image Width",
                    interactive=True,
                )
                num_images = gr.Slider(
                    1,
                    50,
                    value=app_settings.settings.lcm_diffusion_setting.number_of_images,
                    step=1,
                    label="Number of images to generate",
                    interactive=True,
                )
                guidance_scale = gr.Slider(
                    1.0,
                    2.0,
                    value=app_settings.settings.lcm_diffusion_setting.guidance_scale,
                    step=0.1,
                    label="Guidance Scale",
                    interactive=True,
                )

                seed = gr.Slider(
                    value=app_settings.settings.lcm_diffusion_setting.seed,
                    minimum=0,
                    maximum=999999999,
                    label="Seed",
                    step=1,
                    interactive=True,
                )
                seed_checkbox = gr.Checkbox(
                    label="Use seed",
                    value=app_settings.settings.lcm_diffusion_setting.use_seed,
                    interactive=True,
                )

                safety_checker_checkbox = gr.Checkbox(
                    label="Use Safety Checker",
                    value=app_settings.settings.lcm_diffusion_setting.use_safety_checker,
                    interactive=True,
                )
                tiny_auto_encoder_checkbox = gr.Checkbox(
                    label="Use tiny auto encoder for SD",
                    value=app_settings.settings.lcm_diffusion_setting.use_tiny_auto_encoder,
                    interactive=True,
                )
                offline_checkbox = gr.Checkbox(
                    label="Use locally cached model or downloaded model folder(offline)",
                    value=app_settings.settings.lcm_diffusion_setting.use_offline_model,
                    interactive=True,
                )
                img_format = gr.Radio(
                    label="Output image format",
                    choices=["PNG", "JPEG"],
                    value=app_settings.settings.generated_images.format,
                    interactive=True,
                )

        num_inference_steps.change(on_change_inference_steps, num_inference_steps)
        image_height.change(on_change_image_height, image_height)
        image_width.change(on_change_image_width, image_width)
        num_images.change(on_change_num_images, num_images)
        guidance_scale.change(on_change_guidance_scale, guidance_scale)
        seed.change(on_change_seed_value, seed)
        seed_checkbox.change(on_change_seed_checkbox, seed_checkbox)
        safety_checker_checkbox.change(
            on_change_safety_checker_checkbox, safety_checker_checkbox
        )
        tiny_auto_encoder_checkbox.change(
            on_change_tiny_auto_encoder_checkbox, tiny_auto_encoder_checkbox
        )
        offline_checkbox.change(on_offline_checkbox, offline_checkbox)
        img_format.change(on_change_image_format, img_format)
