from app_settings import AppSettings
from typing import Any
import gradio as gr
from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO
from state import get_settings

app_settings = get_settings()


def get_models_ui() -> None:
    global app_settings

    def change_lcm_model_id(model_id):
        app_settings.settings.lcm_diffusion_setting.lcm_model_id = model_id

    def change_lcm_lora_model_id(model_id):
        app_settings.settings.lcm_diffusion_setting.lcm_lora.lcm_lora_id = model_id

    def change_lcm_lora_base_model_id(model_id):
        app_settings.settings.lcm_diffusion_setting.lcm_lora.base_model_id = model_id

    def change_openvino_lcm_model_id(model_id):
        app_settings.settings.lcm_diffusion_setting.openvino_lcm_model_id = model_id

    with gr.Blocks():
        with gr.Row():
            lcm_model_id = gr.Dropdown(
                app_settings.lcm_models,
                label="LCM model ID",
                info="Pytorch LCM model ID",
                value=LCM_DEFAULT_MODEL,
                interactive=True,
            )
        with gr.Row():
            lcm_lora_model_id = gr.Dropdown(
                app_settings.lcm_lora_models,
                label="LCM LoRA model ID",
                info="Pytorch LCM LoRA model ID",
                value="latent-consistency/lcm-lora-sdv1-5",
                interactive=True,
            )
            lcm_lora_base_model_id = gr.Dropdown(
                app_settings.stable_diffsuion_models,
                label="LCM loRA base model ID",
                info="Pytorch LCM LoRA base model ID",
                value="Lykon/dreamshaper-8",
                interactive=True,
            )
        with gr.Row():
            lcm_openvino_model_id = gr.Dropdown(
                app_settings.openvino_lcm_models,
                label="LCM OpenVINO model ID",
                info="OpenVINO LCM-LoRA fused model ID",
                value=LCM_DEFAULT_MODEL_OPENVINO,
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
