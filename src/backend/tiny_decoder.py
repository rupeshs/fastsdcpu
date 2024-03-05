from constants import (
    TAESD_MODEL,
    TAESDXL_MODEL,
    TAESD_MODEL_OPENVINO,
    TAESDXL_MODEL_OPENVINO,
)


def get_tiny_decoder_vae_model(pipeline_class) -> str:
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
    elif pipeline_class == "OVStableDiffusionXLPipeline":
        return TAESDXL_MODEL_OPENVINO
    else:
        raise Exception("No valid pipeline class found!")
