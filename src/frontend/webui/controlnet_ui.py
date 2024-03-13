import gradio as gr
from os import path
from PIL import Image
from backend.lora import get_lora_models
from state import get_settings
from models.interface_types import InterfaceType
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting, ControlNetSetting


_controlnet_models_map = None
_controlnet_enabled = False

app_settings = get_settings()


def on_user_input(
    enable: bool, adapter_name: str, conditioning_scale: float, control_image: Image
):
    if app_settings.settings.lcm_diffusion_setting.controlnet == None:
        app_settings.settings.lcm_diffusion_setting.controlnet = ControlNetSetting()

    if enable and (adapter_name == None or adapter_name == ""):
        gr.Warning("Please select a valid ControlNet adapter")
        return gr.Checkbox.update(value=False)
    elif enable and not control_image:
        gr.Warning("Please provide a ControlNet control image")
        return gr.Checkbox.update(value=False)

    if enable == False:
        app_settings.settings.lcm_diffusion_setting.controlnet.enabled = False
    else:
        app_settings.settings.lcm_diffusion_setting.controlnet.enabled = True
        app_settings.settings.lcm_diffusion_setting.controlnet.adapter_path = (
            _controlnet_models_map[adapter_name]
        )
        app_settings.settings.lcm_diffusion_setting.controlnet.conditioning_scale = (
            conditioning_scale
        )
        app_settings.settings.lcm_diffusion_setting.controlnet._control_image = (
            control_image
        )

    global _controlnet_enabled
    if (
        app_settings.settings.lcm_diffusion_setting.controlnet.enabled
        != _controlnet_enabled
    ):
        app_settings.settings.lcm_diffusion_setting.rebuild_pipeline = True
        _controlnet_enabled = not _controlnet_enabled
    return gr.Checkbox.update(value=enable)


def get_controlnet_ui() -> None:
    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    global _controlnet_models_map
                    _controlnet_models_map = get_lora_models(
                        app_settings.settings.lcm_diffusion_setting.dirs["controlnet"]
                    )

                    enabled_checkbox = gr.Checkbox(
                        label="Enable ControlNet",
                        show_label=True,
                    )
                    model_dropdown = gr.Dropdown(
                        _controlnet_models_map.keys(),
                        label="ControlNet model",
                        info="ControlNet model to load (.safetensors format)",
                        # value=valid_model,
                        interactive=True,
                    )
                    conditioning_scale_slider = gr.Slider(
                        0.0,
                        1.0,
                        value=0.5,
                        step=0.05,
                        label="ControlNet conditioning scale",
                        interactive=True,
                    )
                    control_image = gr.Image(
                        label="Control image",
                        type="pil",
                    )

    enabled_checkbox.input(
        fn=on_user_input,
        inputs=[
            enabled_checkbox,
            model_dropdown,
            conditioning_scale_slider,
            control_image,
        ],
        outputs=[enabled_checkbox],
    )
    model_dropdown.input(
        fn=on_user_input,
        inputs=[
            enabled_checkbox,
            model_dropdown,
            conditioning_scale_slider,
            control_image,
        ],
        outputs=[enabled_checkbox],
    )
    conditioning_scale_slider.input(
        fn=on_user_input,
        inputs=[
            enabled_checkbox,
            model_dropdown,
            conditioning_scale_slider,
            control_image,
        ],
        outputs=[enabled_checkbox],
    )
    control_image.change(
        fn=on_user_input,
        inputs=[
            enabled_checkbox,
            model_dropdown,
            conditioning_scale_slider,
            control_image,
        ],
        outputs=[enabled_checkbox],
    )
