import gradio as gr
from os import path
from backend.lora import (
    get_lora_models,
    get_active_lora_weights,
    update_lora_weights,
    load_lora_weight,
)
from state import get_settings, get_context
from frontend.utils import get_valid_lora_model
from models.interface_types import InterfaceType
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting


_MAX_LORA_WEIGHTS = 5

_custom_lora_sliders = []
_custom_lora_names = []
_custom_lora_columns = []

app_settings = get_settings()


def on_click_update_weight(*lora_weights):
    update_weights = []
    active_weights = get_active_lora_weights()
    if not len(active_weights):
        gr.Warning("No active LoRAs, first you need to load LoRA model")
        return
    for idx, lora in enumerate(active_weights):
        update_weights.append(
            (
                lora[0],
                lora_weights[idx],
            )
        )
    if len(update_weights) > 0:
        update_lora_weights(
            get_context(InterfaceType.WEBUI).lcm_text_to_image.pipeline,
            app_settings.settings.lcm_diffusion_setting,
            update_weights,
        )


def on_click_load_lora(lora_name, lora_weight):
    if app_settings.settings.lcm_diffusion_setting.use_openvino:
        gr.Warning("Currently LoRA is not supported in OpenVINO.")
        return
    lora_models_map = get_lora_models(
        app_settings.settings.lcm_diffusion_setting.lora.models_dir
    )

    # Load a new LoRA
    settings = app_settings.settings.lcm_diffusion_setting
    settings.lora.fuse = False
    settings.lora.enabled = False
    settings.lora.path = lora_models_map[lora_name]
    settings.lora.weight = lora_weight
    if not path.exists(settings.lora.path):
        gr.Warning("Invalid LoRA model path!")
        return
    pipeline = get_context(InterfaceType.WEBUI).lcm_text_to_image.pipeline
    if not pipeline:
        gr.Warning("Pipeline not initialized. Please generate an image first.")
        return
    settings.lora.enabled = True
    load_lora_weight(
        get_context(InterfaceType.WEBUI).lcm_text_to_image.pipeline,
        settings,
    )

    # Update Gradio LoRA UI
    global _MAX_LORA_WEIGHTS
    values = []
    labels = []
    rows = []
    active_weights = get_active_lora_weights()
    for idx, lora in enumerate(active_weights):
        labels.append(f"{lora[0]}: ")
        values.append(lora[1])
        rows.append(gr.Row.update(visible=True))
    for i in range(len(active_weights), _MAX_LORA_WEIGHTS):
        labels.append(f"Update weight")
        values.append(0.0)
        rows.append(gr.Row.update(visible=False))
    return labels + values + rows


def get_lora_models_ui() -> None:
    with gr.Blocks() as ui:
        gr.HTML(
            "Download and place your LoRA model weights in <b>lora_models</b> folders and restart App"
        )
        with gr.Row():

            with gr.Column():
                with gr.Row():
                    lora_models_map = get_lora_models(
                        app_settings.settings.lcm_diffusion_setting.lora.models_dir
                    )
                    valid_model = get_valid_lora_model(
                        list(lora_models_map.values()),
                        app_settings.settings.lcm_diffusion_setting.lora.path,
                        app_settings.settings.lcm_diffusion_setting.lora.models_dir,
                    )
                    if valid_model != "":
                        valid_model_path = lora_models_map[valid_model]
                        app_settings.settings.lcm_diffusion_setting.lora.path = (
                            valid_model_path
                        )
                    else:
                        app_settings.settings.lcm_diffusion_setting.lora.path = ""

                    lora_model = gr.Dropdown(
                        lora_models_map.keys(),
                        label="LoRA model",
                        info="LoRA model weight to load (You can use Lora models from Civitai or Hugging Face .safetensors format)",
                        value=valid_model,
                        interactive=True,
                    )

                    lora_weight = gr.Slider(
                        0.0,
                        1.0,
                        value=app_settings.settings.lcm_diffusion_setting.lora.weight,
                        step=0.05,
                        label="Initial Lora weight",
                        interactive=True,
                    )
                    load_lora_btn = gr.Button(
                        "Load selected LoRA",
                        elem_id="load_lora_button",
                        scale=0,
                    )

                with gr.Row():
                    gr.Markdown(
                        "## Loaded LoRA models",
                        show_label=False,
                    )
                    update_lora_weights_btn = gr.Button(
                        "Update LoRA weights",
                        elem_id="load_lora_button",
                        scale=0,
                    )

                global _MAX_LORA_WEIGHTS
                global _custom_lora_sliders
                global _custom_lora_names
                global _custom_lora_columns
                for i in range(0, _MAX_LORA_WEIGHTS):
                    new_row = gr.Column(visible=False)
                    _custom_lora_columns.append(new_row)
                    with new_row:
                        lora_name = gr.Markdown(
                            "Lora Name",
                            show_label=True,
                        )
                        lora_slider = gr.Slider(
                            0.0,
                            1.0,
                            step=0.05,
                            label="LoRA weight",
                            interactive=True,
                            visible=True,
                        )

                        _custom_lora_names.append(lora_name)
                        _custom_lora_sliders.append(lora_slider)

    load_lora_btn.click(
        fn=on_click_load_lora,
        inputs=[lora_model, lora_weight],
        outputs=[
            *_custom_lora_names,
            *_custom_lora_sliders,
            *_custom_lora_columns,
        ],
    )

    update_lora_weights_btn.click(
        fn=on_click_update_weight,
        inputs=[*_custom_lora_sliders],
        outputs=None,
    )
