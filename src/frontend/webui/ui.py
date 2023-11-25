import gradio as gr
from constants import APP_VERSION
from frontend.webui.text_to_image_ui import get_text_to_image_ui
from frontend.webui.image_to_image_ui import get_image_to_image_ui
from frontend.webui.models_ui import get_models_ui
from paths import FastStableDiffusionPaths
from app_settings import AppSettings


def _get_footer_message() -> str:
    version = f"<center><p> {APP_VERSION} "
    footer_msg = version + (
        '  © 2023 <a href="https://github.com/rupeshs">'
        " Rupesh Sreeraman</a></p></center>"
    )
    return footer_msg


_app_settings = None


def get_web_ui(app_settings: AppSettings) -> gr.Blocks:
    global _app_settings
    _app_settings = app_settings

    def change_mode(mode):
        global _app_settings
        print(mode)
        print(_app_settings)
        _app_settings.settings.lcm_diffusion_setting.use_lcm_lora = False
        _app_settings.settings.lcm_diffusion_setting.use_openvino = False
        if mode == "LCM-LoRA":
            _app_settings.settings.lcm_diffusion_setting.use_lcm_lora = True
        elif mode == "LCM-OpenVINO":
            _app_settings.settings.lcm_diffusion_setting.use_openvino = True

    with gr.Blocks(
        css=FastStableDiffusionPaths.get_css_path(),
        title="FastSD CPU",
    ) as fastsd_web_ui:
        gr.HTML("<center><H1>FastSD CPU</H1></center>")
        mode = gr.Radio(
            ["LCM", "LCM-LoRA", "LCM-OpenVINO"],
            label="Mode",
            info="Current working mode",
            value="LCM",
        )
        mode.change(change_mode, inputs=mode)

        with gr.Tabs():
            with gr.TabItem("Text to Image"):
                get_text_to_image_ui(_app_settings)
            with gr.TabItem("Image to Image"):
                get_image_to_image_ui(_app_settings)
            with gr.TabItem("Models"):
                get_models_ui(_app_settings)

        gr.HTML(_get_footer_message())

    return fastsd_web_ui


def start_webui(
    app_settings: AppSettings,
    share: bool = False,
):
    webui = get_web_ui(app_settings)
    webui.launch(share=share)
