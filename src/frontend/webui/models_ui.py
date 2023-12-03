from app_settings import AppSettings
from typing import Any
import gradio as gr
from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO
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

        lcm_model_id.change(change_lcm_model_id, lcm_model_id)
        lcm_lora_model_id.change(change_lcm_lora_model_id, lcm_lora_model_id)
        lcm_lora_base_model_id.change(
            change_lcm_lora_base_model_id, lcm_lora_base_model_id
        )
        lcm_openvino_model_id.change(
            change_openvino_lcm_model_id, lcm_openvino_model_id
        )
