from diffusers import DiffusionPipeline, LCMScheduler
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
import torch
import os.path


def get_lcm_lora_pipeline(
    base_model_id: str,
    lcm_lora_id: str,
    use_local_model: bool,
    torch_data_type: torch.dtype,
    lcm_diffusion_setting: LCMDiffusionSetting = None,
):
    pipeline = DiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch_data_type,
        local_files_only=use_local_model,
    )
    kwargs = {
        "local_files_only": use_local_model,
        "weight_name": "pytorch_lora_weights.safetensors",
    }
    pipeline.load_lora_weights(
        lcm_lora_id,
        **kwargs,
        adapter_name="lcm",
    )
    if lcm_diffusion_setting.lora_path:
        lora_dir = os.path.dirname(lcm_diffusion_setting.lora_path)
        lora_name = os.path.basename(lcm_diffusion_setting.lora_path)
        adapter_name = os.path.splitext(lora_name)[0]
        pipeline.load_lora_weights(
            lora_dir,
            weight_name=lora_name,
            local_files_only=True,
            adapter_name=adapter_name,
        )
        pipeline.set_adapters(
            ["lcm", adapter_name],
            adapter_weights=[1.0, lcm_diffusion_setting.lora_weight],
        )

    if "lcm" in lcm_lora_id.lower():
        print("LCM LoRA model detected so using recommended LCMScheduler")
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    if lcm_diffusion_setting.fuse_lora:
        pipeline.fuse_lora()
    pipeline.unet.to(memory_format=torch.channels_last)
    return pipeline
