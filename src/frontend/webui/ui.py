import gradio as gr
from constants import APP_VERSION
from frontend.webui.text_to_image_ui import get_text_to_image_ui
from frontend.webui.image_to_image_ui import get_image_to_image_ui
from frontend.webui.generation_settings_ui import get_generation_settings_ui
from frontend.webui.models_ui import get_models_ui
from frontend.webui.image_variations_ui import get_image_variations_ui
from frontend.webui.upscaler_ui import get_upscaler_ui
from frontend.webui.lora_models_ui import get_lora_models_ui
from frontend.webui.controlnet_ui import get_controlnet_ui
from paths import FastStableDiffusionPaths
from state import get_settings

app_settings = get_settings()


def _get_footer_message() -> str:
    version = f"<center><p> {APP_VERSION} "
    footer_msg = version + (
        '  © 2023 - 2024 <a href="https://github.com/rupeshs">'
        " Rupesh Sreeraman</a></p></center>"
    )
    return footer_msg


def get_web_ui() -> gr.Blocks:
    def change_mode(mode):
        global app_settings
        app_settings.settings.lcm_diffusion_setting.use_lcm_lora = False
        app_settings.settings.lcm_diffusion_setting.use_openvino = False
        if mode == "LCM-LoRA":
            app_settings.settings.lcm_diffusion_setting.use_lcm_lora = True
        elif mode == "LCM-OpenVINO":
            app_settings.settings.lcm_diffusion_setting.use_openvino = True

    # Prevent saved LoRA and ControlNet settings from being used by
    # default; in WebUI mode, the user must explicitly enable those
    if app_settings.settings.lcm_diffusion_setting.lora:
        app_settings.settings.lcm_diffusion_setting.lora.enabled = False
    if app_settings.settings.lcm_diffusion_setting.controlnet:
        app_settings.settings.lcm_diffusion_setting.controlnet.enabled = False

    with gr.Blocks(
        css=FastStableDiffusionPaths.get_css_path(),
        title="FastSD CPU",
    ) as fastsd_web_ui:
        gr.HTML("<center><H1>FastSD CPU</H1></center>")
        current_mode = "LCM"
        if app_settings.settings.lcm_diffusion_setting.use_openvino:
            current_mode = "LCM-OpenVINO"
        elif app_settings.settings.lcm_diffusion_setting.use_lcm_lora:
            current_mode = "LCM-LoRA"

        mode = gr.Radio(
            ["LCM", "LCM-LoRA", "LCM-OpenVINO"],
            label="Mode",
            info="Current working mode",
            value=current_mode,
        )
        mode.change(change_mode, inputs=mode)

        with gr.Tabs():
            with gr.TabItem("Text to Image"):
                get_text_to_image_ui()
            with gr.TabItem("Image to Image"):
                get_image_to_image_ui()
            with gr.TabItem("Image Variations"):
                get_image_variations_ui()
            with gr.TabItem("Upscaler"):
                get_upscaler_ui()
            with gr.TabItem("Generation Settings"):
                get_generation_settings_ui()
            with gr.TabItem("Models"):
                get_models_ui()
            with gr.TabItem("Lora Models"):
                get_lora_models_ui()
            with gr.TabItem("ControlNet"):
                get_controlnet_ui()

        gr.HTML(_get_footer_message())

    return fastsd_web_ui


def start_webui(
    share: bool = False,
):
    webui = get_web_ui()
    webui.queue()
    webui.launch(share=share)
