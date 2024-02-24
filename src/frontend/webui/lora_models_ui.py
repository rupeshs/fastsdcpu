import gradio as gr
from backend.lora import get_lora_models
from state import get_settings
from frontend.utils import get_valid_lora_model

app_settings = get_settings()


def change_lora_model(lora_name):
    lora_models_map = get_lora_models(
        app_settings.settings.lcm_diffusion_setting.lora.models_dir
    )
    app_settings.settings.lcm_diffusion_setting.lora.path = lora_models_map[lora_name]


def on_change_use_lora_checkbox(use_lora_checkbox):
    app_settings.settings.lcm_diffusion_setting.lora.enabled = use_lora_checkbox


def on_change_lora_weight(lora_weight):
    app_settings.settings.lcm_diffusion_setting.lora.weight = lora_weight


def get_lora_models_ui() -> None:
    with gr.Blocks():

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

                    use_lora_checkbox = gr.Checkbox(
                        label="Use LoRA model",
                        value=app_settings.settings.lcm_diffusion_setting.lora.enabled,
                        interactive=True,
                    )

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
                    step=0.1,
                    label="Lora weight",
                    interactive=True,
                )

        lora_model.change(change_lora_model, lora_model)
        use_lora_checkbox.change(on_change_use_lora_checkbox, use_lora_checkbox)
        lora_weight.change(on_change_lora_weight, lora_weight)
