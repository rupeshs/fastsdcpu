import gradio as gr
from constants import APP_VERSION
from frontend.webui.text_to_image_ui import get_text_to_image_ui
from paths import FastStableDiffusionPaths
from app_settings import AppSettings


def _get_footer_message() -> str:
    version = f"<center><p> v{APP_VERSION} "
    footer_msg = version + (
        '  © 2023 <a href="https://github.com/rupeshs">'
        " Rupesh Sreeraman</a></p></center>"
    )
    return footer_msg


def get_web_ui(app_settings: AppSettings) -> gr.Blocks:
    with gr.Blocks(
        css=FastStableDiffusionPaths.get_css_path(),
        title="FastSD CPU",
    ) as fastsd_web_ui:
        gr.HTML("<center><H1>FastSD CPU</H1></center>")
        with gr.Tabs():
            with gr.TabItem("Text to Image"):
                get_text_to_image_ui(app_settings)
        gr.HTML(_get_footer_message())

    return fastsd_web_ui


def start_webui(
    app_settings: AppSettings,
    share: bool = False,
):
    webui = get_web_ui(app_settings)
    webui.launch(share=share)
