import gradio as gr
from constants import LCM_DEFAULT_MODEL
from state import get_settings
from frontend.utils import get_valid_model_id

app_settings = get_settings()
app_settings.settings.lcm_diffusion_setting.openvino_lcm_model_id = get_valid_model_id(
    app_settings.openvino_lcm_models,
    app_settings.settings.lcm_diffusion_setting.openvino_lcm_model_id,
)


def change_lcm_model_id(model_id):
    app_settings.settings.lcm_diffusion_setting.lcm_model_id = model_id


def change_lcm_lora_model_id(model_id):
    app_settings.settings.lcm_diffusion_setting.lcm_lora.lcm_lora_id = model_id


def change_lcm_lora_base_model_id(model_id):
    app_settings.settings.lcm_diffusion_setting.lcm_lora.base_model_id = model_id


def change_openvino_lcm_model_id(model_id):
    app_settings.settings.lcm_diffusion_setting.openvino_lcm_model_id = model_id


def change_gguf_diffusion_model(model_path):
    if model_path == "None":
        app_settings.settings.lcm_diffusion_setting.gguf_model.diffusion_path = ""
    else:
        app_settings.settings.lcm_diffusion_setting.gguf_model.diffusion_path = (
            model_path
        )


def change_gguf_clip_model(model_path):
    if model_path == "None":
        app_settings.settings.lcm_diffusion_setting.gguf_model.clip_path = ""
    else:
        app_settings.settings.lcm_diffusion_setting.gguf_model.clip_path = model_path


def change_gguf_t5xxl_model(model_path):
    if model_path == "None":
        app_settings.settings.lcm_diffusion_setting.gguf_model.t5xxl_path = ""
    else:
        app_settings.settings.lcm_diffusion_setting.gguf_model.t5xxl_path = model_path


def change_gguf_vae_model(model_path):
    if model_path == "None":
        app_settings.settings.lcm_diffusion_setting.gguf_model.vae_path = ""
    else:
        app_settings.settings.lcm_diffusion_setting.gguf_model.vae_path = model_path


def get_models_ui() -> None:
    with gr.Blocks():
        with gr.Row():
            lcm_model_id = gr.Dropdown(
                app_settings.lcm_models,
                label="LCM model",
                info="Diffusers LCM model ID",
                value=get_valid_model_id(
                    app_settings.lcm_models,
                    app_settings.settings.lcm_diffusion_setting.lcm_model_id,
                    LCM_DEFAULT_MODEL,
                ),
                interactive=True,
            )
        with gr.Row():
            lcm_lora_model_id = gr.Dropdown(
                app_settings.lcm_lora_models,
                label="LCM LoRA model",
                info="Diffusers LCM LoRA model ID",
                value=get_valid_model_id(
                    app_settings.lcm_lora_models,
                    app_settings.settings.lcm_diffusion_setting.lcm_lora.lcm_lora_id,
                ),
                interactive=True,
            )
            lcm_lora_base_model_id = gr.Dropdown(
                app_settings.stable_diffsuion_models,
                label="LCM LoRA base model",
                info="Diffusers LCM LoRA base model ID",
                value=get_valid_model_id(
                    app_settings.stable_diffsuion_models,
                    app_settings.settings.lcm_diffusion_setting.lcm_lora.base_model_id,
                ),
                interactive=True,
            )
        with gr.Row():
            lcm_openvino_model_id = gr.Dropdown(
                app_settings.openvino_lcm_models,
                label="LCM OpenVINO model",
                info="OpenVINO LCM-LoRA fused model ID",
                value=get_valid_model_id(
                    app_settings.openvino_lcm_models,
                    app_settings.settings.lcm_diffusion_setting.openvino_lcm_model_id,
                ),
                interactive=True,
            )
        with gr.Row():
            gguf_diffusion_model_id = gr.Dropdown(
                app_settings.gguf_diffusion_models,
                label="GGUF diffusion model",
                info="GGUF diffusion model ",
                value=get_valid_model_id(
                    app_settings.gguf_diffusion_models,
                    app_settings.settings.lcm_diffusion_setting.gguf_model.diffusion_path,
                ),
                interactive=True,
            )
        with gr.Row():
            gguf_clip_model_id = gr.Dropdown(
                app_settings.gguf_clip_models,
                label="GGUF CLIP model",
                info="GGUF CLIP model ",
                value=get_valid_model_id(
                    app_settings.gguf_clip_models,
                    app_settings.settings.lcm_diffusion_setting.gguf_model.clip_path,
                ),
                interactive=True,
            )
            gguf_t5xxl_model_id = gr.Dropdown(
                app_settings.gguf_t5xxl_models,
                label="GGUF T5-XXL model",
                info="GGUF T5-XXL model ",
                value=get_valid_model_id(
                    app_settings.gguf_t5xxl_models,
                    app_settings.settings.lcm_diffusion_setting.gguf_model.t5xxl_path,
                ),
                interactive=True,
            )
        with gr.Row():
            gguf_vae_model_id = gr.Dropdown(
                app_settings.gguf_vae_models,
                label="GGUF VAE model",
                info="GGUF VAE model ",
                value=get_valid_model_id(
                    app_settings.gguf_vae_models,
                    app_settings.settings.lcm_diffusion_setting.gguf_model.vae_path,
                ),
                interactive=True,
            )

        lcm_model_id.change(
            change_lcm_model_id,
            lcm_model_id,
        )
        lcm_lora_model_id.change(
            change_lcm_lora_model_id,
            lcm_lora_model_id,
        )
        lcm_lora_base_model_id.change(
            change_lcm_lora_base_model_id,
            lcm_lora_base_model_id,
        )
        lcm_openvino_model_id.change(
            change_openvino_lcm_model_id,
            lcm_openvino_model_id,
        )
        gguf_diffusion_model_id.change(
            change_gguf_diffusion_model,
            gguf_diffusion_model_id,
        )
        gguf_clip_model_id.change(
            change_gguf_clip_model,
            gguf_clip_model_id,
        )
        gguf_t5xxl_model_id.change(
            change_gguf_t5xxl_model,
            gguf_t5xxl_model_id,
        )
        gguf_vae_model_id.change(
            change_gguf_vae_model,
            gguf_vae_model_id,
        )
