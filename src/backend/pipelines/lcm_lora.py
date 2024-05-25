import pathlib
from os import path

import torch
from diffusers import (
    AutoPipelineForText2Image,
    LCMScheduler,
    StableDiffusionPipeline,
)


def load_lcm_weights(
    pipeline,
    use_local_model,
    lcm_lora_id,
):
    kwargs = {
        "local_files_only": use_local_model,
        "weight_name": "pytorch_lora_weights.safetensors",
    }
    pipeline.load_lora_weights(
        lcm_lora_id,
        **kwargs,
        adapter_name="lcm",
    )


def get_lcm_lora_pipeline(
    base_model_id: str,
    lcm_lora_id: str,
    use_local_model: bool,
    torch_data_type: torch.dtype,
    pipeline_args={},
):
    if pathlib.Path(base_model_id).suffix == ".safetensors":
        # SD 1.5 models only
        # When loading a .safetensors model, the pipeline has to be created
        # with StableDiffusionPipeline() since it's the only class that
        # defines the method from_single_file(); afterwards a new pipeline
        # is created using AutoPipelineForText2Image() for ControlNet
        # support, in case ControlNet is enabled
        if not path.exists(base_model_id):
            raise FileNotFoundError(
                f"Model file not found,Please check your model path: {base_model_id}"
            )
        print("Using single file Safetensors model (Supported models - SD 1.5 models)")

        dummy_pipeline = StableDiffusionPipeline.from_single_file(
            base_model_id,
            torch_dtype=torch_data_type,
            safety_checker=None,
            load_safety_checker=False,
            local_files_only=use_local_model,
            use_safetensors=True,
        )
        pipeline = AutoPipelineForText2Image.from_pipe(
            dummy_pipeline,
            **pipeline_args,
        )
        del dummy_pipeline
    else:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            base_model_id,
            torch_dtype=torch_data_type,
            local_files_only=use_local_model,
            **pipeline_args,
        )

    load_lcm_weights(
        pipeline,
        use_local_model,
        lcm_lora_id,
    )
    # Always fuse LCM-LoRA
    pipeline.fuse_lora()

    if "lcm" in lcm_lora_id.lower() or "hypersd" in lcm_lora_id.lower():
        print("LCM LoRA model detected so using recommended LCMScheduler")
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

    # pipeline.unet.to(memory_format=torch.channels_last)
    return pipeline
