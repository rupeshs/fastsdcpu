from constants import LCM_DEFAULT_MODEL
from diffusers import (
    DiffusionPipeline,
    AutoencoderTiny,
    UNet2DConditionModel,
    LCMScheduler,
)
import torch
from backend.tiny_decoder import get_tiny_decoder_vae_model
from typing import Any
from diffusers import (
    LCMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    StableDiffusionControlNetPipeline,
)


def _get_lcm_pipeline_from_base_model(
    lcm_model_id: str,
    base_model_id: str,
    use_local_model: bool,
):
    pipeline = None
    unet = UNet2DConditionModel.from_pretrained(
        lcm_model_id,
        torch_dtype=torch.float32,
        local_files_only=use_local_model,
        resume_download=True,
    )
    pipeline = DiffusionPipeline.from_pretrained(
        base_model_id,
        unet=unet,
        torch_dtype=torch.float32,
        local_files_only=use_local_model,
        resume_download=True,
    )
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    return pipeline


def load_taesd(
    pipeline: Any,
    use_local_model: bool = False,
    torch_data_type: torch.dtype = torch.float32,
):
    vae_model = get_tiny_decoder_vae_model(pipeline.__class__.__name__)
    pipeline.vae = AutoencoderTiny.from_pretrained(
        vae_model,
        torch_dtype=torch_data_type,
        local_files_only=use_local_model,
    )


def get_lcm_model_pipeline(
    model_id: str = LCM_DEFAULT_MODEL,
    use_local_model: bool = False,
    pipeline_args={},
):
    pipeline = None
    if model_id == "latent-consistency/lcm-sdxl":
        pipeline = _get_lcm_pipeline_from_base_model(
            model_id,
            "stabilityai/stable-diffusion-xl-base-1.0",
            use_local_model,
        )

    elif model_id == "latent-consistency/lcm-ssd-1b":
        pipeline = _get_lcm_pipeline_from_base_model(
            model_id,
            "segmind/SSD-1B",
            use_local_model,
        )
    else:
        # pipeline = DiffusionPipeline.from_pretrained(
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            local_files_only=use_local_model,
            **pipeline_args,
        )

    return pipeline


def get_image_to_image_pipeline(pipeline: Any) -> Any:
    components = pipeline.components
    pipeline_class = pipeline.__class__.__name__
    if (
        pipeline_class == "LatentConsistencyModelPipeline"
        or pipeline_class == "StableDiffusionPipeline"
    ):
        return StableDiffusionImg2ImgPipeline(**components)
    elif pipeline_class == "StableDiffusionControlNetPipeline":
        return AutoPipelineForImage2Image.from_pipe(pipeline)
    elif pipeline_class == "StableDiffusionXLPipeline":
        return StableDiffusionXLImg2ImgPipeline(**components)
    else:
        raise Exception(f"Unknown pipeline {pipeline_class}")
