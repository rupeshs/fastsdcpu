from constants import (
    TAESD_MODEL,
    TAESDXL_MODEL,
    TAESD_MODEL_OPENVINO,
    TAESDXL_MODEL_OPENVINO,
    TAEF1_MODEL_OPENVINO,
)


def get_tiny_autoencoder_repo_id(pipeline_class) -> str:
    print(f"Pipeline class : {pipeline_class}")
    if (
        pipeline_class == "LatentConsistencyModelPipeline"
        or pipeline_class == "StableDiffusionPipeline"
        or pipeline_class == "StableDiffusionImg2ImgPipeline"
        or pipeline_class == "StableDiffusionControlNetPipeline"
        or pipeline_class == "StableDiffusionControlNetImg2ImgPipeline"
    ):
        return TAESD_MODEL
    elif (
        pipeline_class == "StableDiffusionXLPipeline"
        or pipeline_class == "StableDiffusionXLImg2ImgPipeline"
    ):
        return TAESDXL_MODEL
    elif (
        pipeline_class == "OVStableDiffusionPipeline"
        or pipeline_class == "OVStableDiffusionImg2ImgPipeline"
    ):
        return TAESD_MODEL_OPENVINO
    elif (
        pipeline_class == "OVStableDiffusionXLPipeline"
        or pipeline_class == "OVStableDiffusionXLImg2ImgPipeline"
    ):
        return TAESDXL_MODEL_OPENVINO
    elif pipeline_class == "OVFluxPipeline":
        return TAEF1_MODEL_OPENVINO
    else:
        raise ValueError(
            f"Tiny autoencoder not available for the pipeline class {pipeline_class}!"
        )
