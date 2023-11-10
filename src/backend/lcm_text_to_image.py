from typing import Any
from diffusers import (
    DiffusionPipeline,
    AutoencoderTiny,
    LCMScheduler,
    UNet2DConditionModel,
)
from os import path
import torch
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
import numpy as np
from constants import DEVICE, LCM_DEFAULT_MODEL
from huggingface_hub import model_info


if DEVICE == "cpu":
    from huggingface_hub import snapshot_download
    from optimum.intel.openvino.modeling_diffusion import OVModelVaeDecoder, OVBaseModel
    from backend.lcmdiffusion.pipelines.openvino.lcm_ov_pipeline import (
        OVLatentConsistencyModelPipeline,
    )
    from backend.lcmdiffusion.pipelines.openvino.lcm_scheduler import (
        LCMScheduler as OpenVinoLCMscheduler,
    )

    class CustomOVModelVaeDecoder(OVModelVaeDecoder):
        def __init__(
            self,
            model,
            parent_model,
            ov_config=None,
            model_dir=None,
        ):
            super(OVModelVaeDecoder, self).__init__(
                model,
                parent_model,
                ov_config,
                "vae_decoder",
                model_dir,
            )


class LCMTextToImage:
    def __init__(
        self,
        device: str = "cpu",
    ) -> None:
        self.pipeline = None
        self.use_openvino = False
        self.device = None
        self.previous_model_id = None
        self.previous_use_tae_sd = False

    def _get_lcm_pipeline(
        self,
        lcm_model_id: str,
        base_model_id: str,
    ):
        pipeline = None
        unet = UNet2DConditionModel.from_pretrained(
            lcm_model_id,
            torch_dtype=torch.float32,
            # resume_download=True,
        )
        pipeline = DiffusionPipeline.from_pretrained(
            base_model_id,
            unet=unet,
            torch_dtype=torch.float32,
            # resume_download=True,
        )
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

        return pipeline

    def _get_diffuser_pipeline(
        self,
        model_id: str,
        use_local_model,
    ):
        pipeline = None
        if model_id == LCM_DEFAULT_MODEL:
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                local_files_only=use_local_model,
            )
        elif model_id == "latent-consistency/lcm-sdxl":
            pipeline = self._get_lcm_pipeline(
                model_id,
                "stabilityai/stable-diffusion-xl-base-1.0",
            )

        elif model_id == "latent-consistency/lcm-ssd-1b":
            pipeline = self._get_lcm_pipeline(
                model_id,
                "segmind/SSD-1B",
            )
        return pipeline

    def init(
        self,
        model_id: str,
        use_openvino: bool = False,
        device: str = "cpu",
        use_local_model: bool = False,
        use_tiny_auto_encoder: bool = False,
    ) -> None:
        self.device = device
        self.use_openvino = use_openvino
        if (
            self.pipeline is None
            or self.previous_model_id != model_id
            or self.previous_use_tae_sd != use_tiny_auto_encoder
        ):
            if self.use_openvino and DEVICE == "cpu":
                if self.pipeline:
                    del self.pipeline
                scheduler = OpenVinoLCMscheduler.from_pretrained(
                    model_id,
                    subfolder="scheduler",
                )

                self.pipeline = OVLatentConsistencyModelPipeline.from_pretrained(
                    model_id,
                    scheduler=scheduler,
                    compile=False,
                    local_files_only=use_local_model,
                )

                if use_tiny_auto_encoder:
                    print("Using Tiny Auto Encoder (OpenVINO)")
                    taesd_dir = snapshot_download(
                        repo_id="deinferno/taesd-openvino",
                        local_files_only=use_local_model,
                    )
                    self.pipeline.vae_decoder = CustomOVModelVaeDecoder(
                        model=OVBaseModel.load_model(
                            f"{taesd_dir}/vae_decoder/openvino_model.xml"
                        ),
                        parent_model=self.pipeline,
                        model_dir=taesd_dir,
                    )

            else:
                if self.pipeline:
                    del self.pipeline

                self.pipeline = self._get_diffuser_pipeline(
                    model_id,
                    use_local_model,
                )

                if use_tiny_auto_encoder:
                    print("Using Tiny Auto Encoder")
                    self.pipeline.vae = AutoencoderTiny.from_pretrained(
                        "madebyollin/taesd",
                        torch_dtype=torch.float32,
                        local_files_only=use_local_model,
                    )

                self.pipeline.to(
                    torch_device=self.device,
                    torch_dtype=torch.float32,
                )

            self.previous_model_id = model_id
            self.previous_use_tae_sd = use_tiny_auto_encoder

    def generate(
        self,
        lcm_diffusion_setting: LCMDiffusionSetting,
        reshape: bool = False,
    ) -> Any:
        if lcm_diffusion_setting.use_seed:
            cur_seed = lcm_diffusion_setting.seed
            if self.use_openvino:
                np.random.seed(cur_seed)
            else:
                torch.manual_seed(cur_seed)

        if self.use_openvino and DEVICE == "cpu":
            print("Using OpenVINO")
            if reshape:
                print("Reshape and compile")
                self.pipeline.reshape(
                    batch_size=1,
                    height=lcm_diffusion_setting.image_height,
                    width=lcm_diffusion_setting.image_width,
                    num_images_per_prompt=lcm_diffusion_setting.number_of_images,
                )
                self.pipeline.compile()

        if not lcm_diffusion_setting.use_safety_checker:
            self.pipeline.safety_checker = None

        result_images = self.pipeline(
            prompt=lcm_diffusion_setting.prompt,
            num_inference_steps=lcm_diffusion_setting.inference_steps,
            guidance_scale=lcm_diffusion_setting.guidance_scale,
            width=lcm_diffusion_setting.image_width,
            height=lcm_diffusion_setting.image_height,
            num_images_per_prompt=lcm_diffusion_setting.number_of_images,
        ).images

        return result_images
