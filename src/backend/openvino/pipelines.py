from constants import DEVICE, LCM_DEFAULT_MODEL_OPENVINO
from backend.tiny_decoder import get_tiny_decoder_vae_model
from typing import Any
from backend.device import is_openvino_device
from paths import get_base_folder_name

if is_openvino_device():
    from huggingface_hub import snapshot_download
    from optimum.intel.openvino.modeling_diffusion import OVBaseModel

    from optimum.intel.openvino.modeling_diffusion import (
        OVStableDiffusionPipeline,
        OVStableDiffusionImg2ImgPipeline,
        OVStableDiffusionXLPipeline,
        OVStableDiffusionXLImg2ImgPipeline,
    )
    from backend.openvino.custom_ov_model_vae_decoder import CustomOVModelVaeDecoder


def ov_load_taesd(
    pipeline: Any,
    use_local_model: bool = False,
):
    taesd_dir = snapshot_download(
        repo_id=get_tiny_decoder_vae_model(pipeline.__class__.__name__),
        local_files_only=use_local_model,
    )
    pipeline.vae_decoder = CustomOVModelVaeDecoder(
        model=OVBaseModel.load_model(f"{taesd_dir}/vae_decoder/openvino_model.xml"),
        parent_model=pipeline,
        model_dir=taesd_dir,
    )


def get_ov_text_to_image_pipeline(
    model_id: str = LCM_DEFAULT_MODEL_OPENVINO,
    use_local_model: bool = False,
) -> Any:
    if "xl" in get_base_folder_name(model_id).lower():
        pipeline = OVStableDiffusionXLPipeline.from_pretrained(
            model_id,
            local_files_only=use_local_model,
            ov_config={"CACHE_DIR": ""},
            device=DEVICE.upper(),
        )
    else:
        pipeline = OVStableDiffusionPipeline.from_pretrained(
            model_id,
            local_files_only=use_local_model,
            ov_config={"CACHE_DIR": ""},
            device=DEVICE.upper(),
        )

    return pipeline


def get_ov_image_to_image_pipeline(
    model_id: str = LCM_DEFAULT_MODEL_OPENVINO,
    use_local_model: bool = False,
) -> Any:
    if "xl" in get_base_folder_name(model_id).lower():
        pipeline = OVStableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            local_files_only=use_local_model,
            ov_config={"CACHE_DIR": ""},
            device=DEVICE.upper(),
        )
    else:
        pipeline = OVStableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            local_files_only=use_local_model,
            ov_config={"CACHE_DIR": ""},
            device=DEVICE.upper(),
        )
    return pipeline
