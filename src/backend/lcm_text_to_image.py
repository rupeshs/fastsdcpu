import gc
from math import ceil
from typing import Any

import numpy as np
import torch
import logging
from backend.device import is_openvino_device
from backend.lora import load_lora_weight
from backend.controlnet import (
    load_controlnet_adapters,
    update_controlnet_arguments,
)
from backend.models.lcmdiffusion_setting import (
    DiffusionTask,
    LCMDiffusionSetting,
    LCMLora,
)
from backend.openvino.pipelines import (
    get_ov_image_to_image_pipeline,
    get_ov_text_to_image_pipeline,
    ov_load_taesd,
)
from backend.pipelines.lcm import (
    get_image_to_image_pipeline,
    get_lcm_model_pipeline,
    load_taesd,
)
from backend.pipelines.lcm_lora import get_lcm_lora_pipeline
from constants import DEVICE
from diffusers import LCMScheduler
from image_ops import resize_pil_image


class LCMTextToImage:
    def __init__(
        self,
        device: str = "cpu",
    ) -> None:
        self.pipeline = None
        self.use_openvino = False
        self.device = ""
        self.previous_model_id = None
        self.previous_use_tae_sd = False
        self.previous_use_lcm_lora = False
        self.previous_ov_model_id = ""
        self.previous_safety_checker = False
        self.previous_use_openvino = False
        self.img_to_img_pipeline = None
        self.is_openvino_init = False
        self.previous_lora = None
        self.task_type = DiffusionTask.text_to_image
        self.torch_data_type = (
            torch.float32 if is_openvino_device() or DEVICE == "mps" else torch.float16
        )
        print(f"Torch datatype : {self.torch_data_type}")

    def _pipeline_to_device(self):
        print(f"Pipeline device : {DEVICE}")
        print(f"Pipeline dtype : {self.torch_data_type}")
        self.pipeline.to(
            torch_device=DEVICE,
            torch_dtype=self.torch_data_type,
        )

    def _add_freeu(self):
        pipeline_class = self.pipeline.__class__.__name__
        if isinstance(self.pipeline.scheduler, LCMScheduler):
            if pipeline_class == "StableDiffusionPipeline":
                print("Add FreeU - SD")
                self.pipeline.enable_freeu(
                    s1=0.9,
                    s2=0.2,
                    b1=1.2,
                    b2=1.4,
                )
            elif pipeline_class == "StableDiffusionXLPipeline":
                print("Add FreeU - SDXL")
                self.pipeline.enable_freeu(
                    s1=0.6,
                    s2=0.4,
                    b1=1.1,
                    b2=1.2,
                )

    def _update_lcm_scheduler_params(self):
        if isinstance(self.pipeline.scheduler, LCMScheduler):
            self.pipeline.scheduler = LCMScheduler.from_config(
                self.pipeline.scheduler.config,
                beta_start=0.001,
                beta_end=0.01,
            )

    def init(
        self,
        device: str = "cpu",
        lcm_diffusion_setting: LCMDiffusionSetting = LCMDiffusionSetting(),
    ) -> None:
        self.device = device
        self.use_openvino = lcm_diffusion_setting.use_openvino
        model_id = lcm_diffusion_setting.lcm_model_id
        use_local_model = lcm_diffusion_setting.use_offline_model
        use_tiny_auto_encoder = lcm_diffusion_setting.use_tiny_auto_encoder
        use_lora = lcm_diffusion_setting.use_lcm_lora
        lcm_lora: LCMLora = lcm_diffusion_setting.lcm_lora
        ov_model_id = lcm_diffusion_setting.openvino_lcm_model_id

        if lcm_diffusion_setting.diffusion_task == DiffusionTask.image_to_image.value:
            lcm_diffusion_setting.init_image = resize_pil_image(
                lcm_diffusion_setting.init_image,
                lcm_diffusion_setting.image_width,
                lcm_diffusion_setting.image_height,
            )

        if (
            self.pipeline is None
            or self.previous_model_id != model_id
            or self.previous_use_tae_sd != use_tiny_auto_encoder
            or self.previous_lcm_lora_base_id != lcm_lora.base_model_id
            or self.previous_lcm_lora_id != lcm_lora.lcm_lora_id
            or self.previous_use_lcm_lora != use_lora
            or self.previous_ov_model_id != ov_model_id
            or self.previous_safety_checker != lcm_diffusion_setting.use_safety_checker
            or self.previous_use_openvino != lcm_diffusion_setting.use_openvino
            or (
                self.use_openvino
                and (
                    self.previous_task_type != lcm_diffusion_setting.diffusion_task
                    or self.previous_lora != lcm_diffusion_setting.lora
                )
            )
            or lcm_diffusion_setting.rebuild_pipeline
        ):
            if self.use_openvino and is_openvino_device():
                if self.pipeline:
                    del self.pipeline
                    self.pipeline = None
                    gc.collect()
                self.is_openvino_init = True
                if (
                    lcm_diffusion_setting.diffusion_task
                    == DiffusionTask.text_to_image.value
                ):
                    print(f"***** Init Text to image (OpenVINO) - {ov_model_id} *****")
                    self.pipeline = get_ov_text_to_image_pipeline(
                        ov_model_id,
                        use_local_model,
                    )
                elif (
                    lcm_diffusion_setting.diffusion_task
                    == DiffusionTask.image_to_image.value
                ):
                    print(f"***** Image to image (OpenVINO) - {ov_model_id} *****")
                    self.pipeline = get_ov_image_to_image_pipeline(
                        ov_model_id,
                        use_local_model,
                    )
            else:
                if self.pipeline:
                    del self.pipeline
                    self.pipeline = None
                if self.img_to_img_pipeline:
                    del self.img_to_img_pipeline
                    self.img_to_img_pipeline = None

                controlnet_args = load_controlnet_adapters(lcm_diffusion_setting)
                if use_lora:
                    print(
                        f"***** Init LCM-LoRA pipeline - {lcm_lora.base_model_id} *****"
                    )
                    self.pipeline = get_lcm_lora_pipeline(
                        lcm_lora.base_model_id,
                        lcm_lora.lcm_lora_id,
                        use_local_model,
                        torch_data_type=self.torch_data_type,
                        pipeline_args=controlnet_args,
                    )

                else:
                    print(f"***** Init LCM Model pipeline - {model_id} *****")
                    self.pipeline = get_lcm_model_pipeline(
                        model_id,
                        use_local_model,
                        controlnet_args,
                    )

                self.img_to_img_pipeline = get_image_to_image_pipeline(self.pipeline)

            if use_tiny_auto_encoder:
                if self.use_openvino and is_openvino_device():
                    print("Using Tiny Auto Encoder (OpenVINO)")
                    ov_load_taesd(
                        self.pipeline,
                        use_local_model,
                    )
                else:
                    print("Using Tiny Auto Encoder")
                    load_taesd(
                        self.pipeline,
                        use_local_model,
                        self.torch_data_type,
                    )
                    load_taesd(
                        self.img_to_img_pipeline,
                        use_local_model,
                        self.torch_data_type,
                    )

            if not self.use_openvino and not is_openvino_device():
                self._pipeline_to_device()

            if (
                lcm_diffusion_setting.diffusion_task
                == DiffusionTask.image_to_image.value
                and lcm_diffusion_setting.use_openvino
            ):
                self.pipeline.scheduler = LCMScheduler.from_config(
                    self.pipeline.scheduler.config,
                )
            else:
                self._update_lcm_scheduler_params()

            if use_lora:
                self._add_freeu()

            self.previous_model_id = model_id
            self.previous_ov_model_id = ov_model_id
            self.previous_use_tae_sd = use_tiny_auto_encoder
            self.previous_lcm_lora_base_id = lcm_lora.base_model_id
            self.previous_lcm_lora_id = lcm_lora.lcm_lora_id
            self.previous_use_lcm_lora = use_lora
            self.previous_safety_checker = lcm_diffusion_setting.use_safety_checker
            self.previous_use_openvino = lcm_diffusion_setting.use_openvino
            self.previous_task_type = lcm_diffusion_setting.diffusion_task
            self.previous_lora = lcm_diffusion_setting.lora.model_copy(deep=True)
            lcm_diffusion_setting.rebuild_pipeline = False
            if (
                lcm_diffusion_setting.diffusion_task
                == DiffusionTask.text_to_image.value
            ):
                print(f"Pipeline : {self.pipeline}")
            elif (
                lcm_diffusion_setting.diffusion_task
                == DiffusionTask.image_to_image.value
            ):
                if self.use_openvino and is_openvino_device():
                    print(f"Pipeline : {self.pipeline}")
                else:
                    print(f"Pipeline : {self.img_to_img_pipeline}")
            if self.use_openvino:
                if lcm_diffusion_setting.lora.enabled:
                    print("Warning: Lora models not supported on OpenVINO mode")
            else:
                adapters = self.pipeline.get_active_adapters()
                print(f"Active adapters : {adapters}")

    def _get_timesteps(self):
        time_steps = self.pipeline.scheduler.config.get("timesteps")
        time_steps_value = [int(time_steps)] if time_steps else None
        return time_steps_value

    def generate(
        self,
        lcm_diffusion_setting: LCMDiffusionSetting,
        reshape: bool = False,
    ) -> Any:
        guidance_scale = lcm_diffusion_setting.guidance_scale
        img_to_img_inference_steps = lcm_diffusion_setting.inference_steps
        check_step_value = int(
            lcm_diffusion_setting.inference_steps * lcm_diffusion_setting.strength
        )
        if (
            lcm_diffusion_setting.diffusion_task == DiffusionTask.image_to_image.value
            and check_step_value < 1
        ):
            img_to_img_inference_steps = ceil(1 / lcm_diffusion_setting.strength)
            print(
                f"Strength: {lcm_diffusion_setting.strength},{img_to_img_inference_steps}"
            )

        if lcm_diffusion_setting.use_seed:
            cur_seed = lcm_diffusion_setting.seed
            if self.use_openvino:
                np.random.seed(cur_seed)
            else:
                torch.manual_seed(cur_seed)

        is_openvino_pipe = lcm_diffusion_setting.use_openvino and is_openvino_device()
        if is_openvino_pipe:
            print("Using OpenVINO")
            if reshape and not self.is_openvino_init:
                print("Reshape and compile")
                self.pipeline.reshape(
                    batch_size=-1,
                    height=lcm_diffusion_setting.image_height,
                    width=lcm_diffusion_setting.image_width,
                    num_images_per_prompt=lcm_diffusion_setting.number_of_images,
                )
                self.pipeline.compile()

            if self.is_openvino_init:
                self.is_openvino_init = False

        if not lcm_diffusion_setting.use_safety_checker:
            self.pipeline.safety_checker = None
            if (
                lcm_diffusion_setting.diffusion_task
                == DiffusionTask.image_to_image.value
                and not is_openvino_pipe
            ):
                self.img_to_img_pipeline.safety_checker = None

        if (
            not lcm_diffusion_setting.use_lcm_lora
            and not lcm_diffusion_setting.use_openvino
            and lcm_diffusion_setting.guidance_scale != 1.0
        ):
            print("Not using LCM-LoRA so setting guidance_scale 1.0")
            guidance_scale = 1.0

        controlnet_args = update_controlnet_arguments(lcm_diffusion_setting)
        if lcm_diffusion_setting.use_openvino:
            if (
                lcm_diffusion_setting.diffusion_task
                == DiffusionTask.text_to_image.value
            ):
                result_images = self.pipeline(
                    prompt=lcm_diffusion_setting.prompt,
                    negative_prompt=lcm_diffusion_setting.negative_prompt,
                    num_inference_steps=lcm_diffusion_setting.inference_steps,
                    guidance_scale=guidance_scale,
                    width=lcm_diffusion_setting.image_width,
                    height=lcm_diffusion_setting.image_height,
                    num_images_per_prompt=lcm_diffusion_setting.number_of_images,
                ).images
            elif (
                lcm_diffusion_setting.diffusion_task
                == DiffusionTask.image_to_image.value
            ):
                result_images = self.pipeline(
                    image=lcm_diffusion_setting.init_image,
                    strength=lcm_diffusion_setting.strength,
                    prompt=lcm_diffusion_setting.prompt,
                    negative_prompt=lcm_diffusion_setting.negative_prompt,
                    num_inference_steps=img_to_img_inference_steps * 3,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=lcm_diffusion_setting.number_of_images,
                ).images

        else:
            if (
                lcm_diffusion_setting.diffusion_task
                == DiffusionTask.text_to_image.value
            ):
                result_images = self.pipeline(
                    prompt=lcm_diffusion_setting.prompt,
                    negative_prompt=lcm_diffusion_setting.negative_prompt,
                    num_inference_steps=lcm_diffusion_setting.inference_steps,
                    guidance_scale=guidance_scale,
                    width=lcm_diffusion_setting.image_width,
                    height=lcm_diffusion_setting.image_height,
                    num_images_per_prompt=lcm_diffusion_setting.number_of_images,
                    timesteps=self._get_timesteps(),
                    **controlnet_args,
                ).images

            elif (
                lcm_diffusion_setting.diffusion_task
                == DiffusionTask.image_to_image.value
            ):
                result_images = self.img_to_img_pipeline(
                    image=lcm_diffusion_setting.init_image,
                    strength=lcm_diffusion_setting.strength,
                    prompt=lcm_diffusion_setting.prompt,
                    negative_prompt=lcm_diffusion_setting.negative_prompt,
                    num_inference_steps=img_to_img_inference_steps,
                    guidance_scale=guidance_scale,
                    width=lcm_diffusion_setting.image_width,
                    height=lcm_diffusion_setting.image_height,
                    num_images_per_prompt=lcm_diffusion_setting.number_of_images,
                    **controlnet_args,
                ).images
        return result_images
