from constants import TAESD_MODEL, TAESDXL_MODEL, TAESD_MODEL_OPENVINO


def get_tiny_decoder_vae_model(pipeline_class) -> str:
    print(f"Pipeline class : {pipeline_class}")
    if (
        pipeline_class == "LatentConsistencyModelPipeline"
        or pipeline_class == "StableDiffusionPipeline"
    ):
        return TAESD_MODEL
    elif pipeline_class == "StableDiffusionXLPipeline":
        return TAESDXL_MODEL
    elif pipeline_class == "OVStableDiffusionPipeline":
        return TAESD_MODEL_OPENVINO
