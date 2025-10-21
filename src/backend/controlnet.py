import logging
from PIL import Image
from typing import Any
from diffusers import (
    ControlNetModel,
    AutoPipelineForText2Image,
    LCMScheduler,
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from backend.models.lcmdiffusion_setting import (
    DiffusionTask,
    ControlNetSetting,
)


# Prepares ControlNet adapters for use with FastSD CPU
#
# This function loads the ControlNet adapters defined by the
# _lcm_diffusion_setting.controlnet_ object and returns a dictionary
# with the pipeline arguments required to use the loaded adapters
def load_controlnet_adapters(lcm_diffusion_setting) -> dict:
    controlnet_args = {}
    if (
        lcm_diffusion_setting.controlnet is None
        or not lcm_diffusion_setting.controlnet.enabled
    ):
        return controlnet_args

    logging.info("Loading ControlNet adapter")
    controlnet_adapter = ControlNetModel.from_single_file(
        lcm_diffusion_setting.controlnet.adapter_path,
        # local_files_only=True,
        use_safetensors=True,
    )
    controlnet_args["controlnet"] = controlnet_adapter
    return controlnet_args


# Updates the ControlNet pipeline arguments to use for image generation
#
# This function uses the contents of the _lcm_diffusion_setting.controlnet_
# object to generate a dictionary with the corresponding pipeline arguments
# to be used for image generation; in particular, it sets the ControlNet control
# image and conditioning scale
def update_controlnet_arguments(lcm_diffusion_setting) -> dict:
    controlnet_args = {}
    if (
        lcm_diffusion_setting.controlnet is None
        or not lcm_diffusion_setting.controlnet.enabled
    ):
        return controlnet_args

    controlnet_args["controlnet_conditioning_scale"] = (
        lcm_diffusion_setting.controlnet.conditioning_scale
    )
    if lcm_diffusion_setting.diffusion_task == DiffusionTask.text_to_image.value:
        controlnet_args["image"] = lcm_diffusion_setting.controlnet._control_image
    elif lcm_diffusion_setting.diffusion_task == DiffusionTask.image_to_image.value:
        controlnet_args["control_image"] = (
            lcm_diffusion_setting.controlnet._control_image
        )
    return controlnet_args


# Helper function to adjust ControlNet settings from a dictionary
def controlnet_settings_from_dict(
    lcm_diffusion_setting,
    dictionary,
) -> None:
    if lcm_diffusion_setting is None or dictionary is None:
        logging.error("Invalid arguments!")
        return
    if (
        "controlnet" not in dictionary
        or dictionary["controlnet"] is None
        or len(dictionary["controlnet"]) == 0
    ):
        logging.warning("ControlNet settings not found, ControlNet will be disabled")
        lcm_diffusion_setting.controlnet = None
        return

    controlnet = ControlNetSetting()
    controlnet.enabled = dictionary["controlnet"][0]["enabled"]
    controlnet.conditioning_scale = dictionary["controlnet"][0]["conditioning_scale"]
    controlnet.adapter_path = dictionary["controlnet"][0]["adapter_path"]
    controlnet._control_image = None
    image_path = dictionary["controlnet"][0]["control_image"]
    if controlnet.enabled:
        try:
            controlnet._control_image = Image.open(image_path)
        except (AttributeError, FileNotFoundError) as err:
            print(err)
        if controlnet._control_image is None:
            logging.error("Wrong ControlNet control image! Disabling ControlNet")
            controlnet.enabled = False
    lcm_diffusion_setting.controlnet = controlnet


def get_controlnet_pipeline(
    pipeline: Any, lcm_diffusion_setting, diffusion_task: DiffusionTask
) -> Any:
    """Creates a ControlNet pipeline from the base txt2img _pipeline_"""
    if (
        lcm_diffusion_setting.controlnet is None
        or not lcm_diffusion_setting.controlnet.enabled
    ):
        return None
    components = pipeline.components
    pipeline_class = pipeline.__class__.__name__
    controlnet_args = load_controlnet_adapters(lcm_diffusion_setting)
    if diffusion_task == DiffusionTask.text_to_image.value:
        if (
            pipeline_class == "LatentConsistencyModelPipeline"
            or pipeline_class == "StableDiffusionPipeline"
        ):
            controlnet_pipeline = StableDiffusionControlNetPipeline.from_pipe(
                pipeline,
                vae=None,
                **controlnet_args,
            )
            controlnet_pipeline.vae = pipeline.vae
            return controlnet_pipeline
        elif pipeline_class == "StableDiffusionXLPipeline":
            controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_pipe(
                pipeline,
                vae=None,
                **controlnet_args,
            )
            controlnet_pipeline.vae = pipeline.vae
            return controlnet_pipeline
    elif diffusion_task == DiffusionTask.image_to_image.value:
        if (
            pipeline_class == "LatentConsistencyModelPipeline"
            or pipeline_class == "StableDiffusionPipeline"
        ):
            controlnet_pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pipe(
                pipeline,
                vae=None,
                **controlnet_args,
            )
            controlnet_pipeline.vae = pipeline.vae
            return controlnet_pipeline
        elif pipeline_class == "StableDiffusionXLPipeline":
            controlnet_pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(
                pipeline,
                vae=None,
                **controlnet_args,
            )
            controlnet_pipeline.vae = pipeline.vae
            return controlnet_pipeline
