from diffusers import DiffusionPipeline, LCMScheduler
import torch


def get_lcm_lora_pipeline(
    base_model_id: str,
    lcm_lora_id: str,
    use_local_model: bool,
    torch_data_type: torch.dtype,
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
    )
    if "lcm" in lcm_lora_id.lower():
        print("LCM LoRA model detected so using recommended LCMScheduler")
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    pipeline.fuse_lora()
    pipeline.unet.to(memory_format=torch.channels_last)
    return pipeline
