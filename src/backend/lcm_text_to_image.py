from typing import Any
from diffusers import DiffusionPipeline
from os import path
from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO
import torch
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
import numpy as np


class LCMTextToImage:
    def __init__(
        self,
    ) -> None:
        self.pipeline = None
        self.use_openvino = False
        self.device = None
        self.previous_model_id = None

    def _get_lcm_diffusion_pipeline_path(self) -> str:
        script_path = path.dirname(path.abspath(__file__))
        file_path = path.join(
            script_path,
            "lcmdiffusion",
            "pipelines",
            "latent_consistency_txt2img.py",
        )
        return file_path

    def init(
        self,
        model_id: str,
        use_openvino: bool = False,
        device: str = "cpu",
        use_local_model: bool = False,
    ) -> None:
        self.device = device
        self.use_openvino = use_openvino
        if self.pipeline is None or self.previous_model_id != model_id:
            if self.use_openvino:
                from backend.lcmdiffusion.pipelines.openvino.lcm_ov_pipeline import (
                    OVLatentConsistencyModelPipeline,
                )

                from backend.lcmdiffusion.pipelines.openvino.lcm_scheduler import (
                    LCMScheduler,
                )

                if self.pipeline:
                    del self.pipeline
                scheduler = LCMScheduler.from_pretrained(
                    model_id,
                    subfolder="scheduler",
                )
                self.pipeline = OVLatentConsistencyModelPipeline.from_pretrained(
                    model_id,
                    scheduler=scheduler,
                    compile=False,
                    local_files_only=use_local_model,
                )
            else:
                if self.pipeline:
                    del self.pipeline

                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    custom_pipeline=self._get_lcm_diffusion_pipeline_path(),
                    custom_revision="main",
                    local_files_only=use_local_model,
                )
                self.pipeline.to(
                    torch_device=self.device,
                    torch_dtype=torch.float32,
                )
            self.previous_model_id = model_id

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

        if self.use_openvino:
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
            lcm_origin_steps=lcm_diffusion_setting.lcm_origin_steps,
            width=lcm_diffusion_setting.image_width,
            height=lcm_diffusion_setting.image_height,
            output_type="pil",
            num_images_per_prompt=lcm_diffusion_setting.number_of_images,
        ).images

        return result_images
