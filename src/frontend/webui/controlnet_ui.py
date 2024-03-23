import gradio as gr
from PIL import Image
from backend.lora import get_lora_models
from state import get_settings
from backend.models.lcmdiffusion_setting import ControlNetSetting
from backend.annotators.image_control_factory import ImageControlFactory

_controlnet_models_map = None
_controlnet_enabled = False
_adapter_path = None

app_settings = get_settings()


def on_user_input(
    enable: bool,
    adapter_name: str,
    conditioning_scale: float,
    control_image: Image,
    preprocessor: str,
):
    if control_image is None:
        return gr.Checkbox.update(value=enable)

    settings = app_settings.settings.lcm_diffusion_setting
    if settings.controlnet == None:
        settings.controlnet = ControlNetSetting()

    if enable and (adapter_name == None or adapter_name == ""):
        gr.Warning("Please select a valid ControlNet adapter")
        return gr.Checkbox.update(value=False)
    elif enable and not control_image:
        gr.Warning("Please provide a ControlNet control image")
        return gr.Checkbox.update(value=False)
    if preprocessor == "None":
        processed_control_image = control_image
    else:
        image_control_factory = ImageControlFactory()
        control = image_control_factory.create_control(preprocessor)
        processed_control_image = control.get_control_image(control_image)

    if not enable:
        settings.controlnet.enabled = False
    else:
        settings.controlnet.enabled = True
        settings.controlnet.adapter_path = _controlnet_models_map[adapter_name]
        settings.controlnet.conditioning_scale = float(conditioning_scale)
        settings.controlnet._control_image = processed_control_image

    # This code can be improved; currently, if the user clicks the
    # "Enable ControlNet" checkbox or changes the currently selected
    # ControlNet model, it will trigger a pipeline rebuild even if, in
    # the end, the user leaves the same ControlNet settings
    global _controlnet_enabled
    global _adapter_path
    if settings.controlnet.enabled != _controlnet_enabled or (
        settings.controlnet.enabled
        and settings.controlnet.adapter_path != _adapter_path
    ):
        settings.rebuild_pipeline = True
        _controlnet_enabled = settings.controlnet.enabled
        _adapter_path = settings.controlnet.adapter_path
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
                        info="Enable ControlNet",
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
                        value=1.0,
                        step=0.05,
                        label="ControlNet conditioning scale",
                        interactive=True,
                    )
                    control_image = gr.Image(
                        label="Control image",
                        type="pil",
                    )
                preprocessor_radio = gr.Radio(
                    ["Canny", "Pose", "None"],
                    label="Preprocessor",
                    info="Select the preprocessor for the control image",
                    value="Canny",
                    interactive=True,
                )

    enabled_checkbox.input(
        fn=on_user_input,
        inputs=[
            enabled_checkbox,
            model_dropdown,
            conditioning_scale_slider,
            control_image,
            preprocessor_radio,
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
            preprocessor_radio,
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
            preprocessor_radio,
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
            preprocessor_radio,
        ],
        outputs=[enabled_checkbox],
    )
    preprocessor_radio.change(
        fn=on_user_input,
        inputs=[
            enabled_checkbox,
            model_dropdown,
            conditioning_scale_slider,
            control_image,
            preprocessor_radio,
        ],
        outputs=[enabled_checkbox],
    )
