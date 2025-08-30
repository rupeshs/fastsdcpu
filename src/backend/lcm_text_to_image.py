import gc
from math import ceil
from typing import Any, List
import random

import numpy as np
import torch
from backend.device import is_openvino_device
from backend.lora import reset_active_lora_weights
from backend.controlnet import (
    load_controlnet_adapters,
    update_controlnet_arguments,
    get_controlnet_pipeline,
)
from backend.models.lcmdiffusion_setting import (
    DiffusionTask,
    LCMDiffusionSetting,
    LCMLora,
)
from backend.openvino.pipelines import (
    get_ov_image_to_image_pipeline,
    get_ov_text_to_image_pipeline,
    ov_load_tiny_autoencoder,
    get_ov_diffusion_pipeline,
)
from backend.pipelines.lcm import (
    get_image_to_image_pipeline,
    get_lcm_model_pipeline,
    load_taesd,
)
from backend.pipelines.lcm_lora import get_lcm_lora_pipeline
from constants import DEVICE, GGUF_THREADS
from diffusers import LCMScheduler
from image_ops import resize_pil_image
from backend.openvino.ov_hc_stablediffusion_pipeline import OvHcLatentConsistency
from backend.gguf.gguf_diffusion import (
    GGUFDiffusion,
    ModelConfig,
    Txt2ImgConfig,
    SampleMethod,
)
from paths import get_app_path
from pprint import pprint

try:
    # support for token merging; keeping it optional for now
    import tomesd
except ImportError:
    print("tomesd library unavailable; disabling token merging support")
    tomesd = None


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
        self.previous_token_merging = 0.0
        self.previous_safety_checker = False
        self.previous_use_openvino = False
        self.img_to_img_pipeline = None
        self.is_openvino_init = False
        self.previous_lora = None
        self.task_type = DiffusionTask.text_to_image
        self.previous_use_gguf_model = False
        self.previous_gguf_model = None
        self.torch_data_type = (
            torch.float32 if is_openvino_device() or DEVICE == "mps" else torch.float16
        )
        self.ov_model_id = None
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

    def _enable_vae_tiling(self):
        self.pipeline.vae.enable_tiling()

    def _update_lcm_scheduler_params(self):
        if isinstance(self.pipeline.scheduler, LCMScheduler):
            self.pipeline.scheduler = LCMScheduler.from_config(
                self.pipeline.scheduler.config,
                beta_start=0.001,
                beta_end=0.01,
            )

    def _is_hetero_pipeline(self) -> bool:
        return "square" in self.ov_model_id.lower()

    def _load_ov_hetero_pipeline(self):
        print("Loading Heterogeneous Compute pipeline")
        if DEVICE.upper() == "NPU":
            device = ["NPU", "NPU", "NPU"]
            self.pipeline = OvHcLatentConsistency(self.ov_model_id, device)
        else:
            self.pipeline = OvHcLatentConsistency(self.ov_model_id)

    def _generate_images_hetero_compute(
        self,
        lcm_diffusion_setting: LCMDiffusionSetting,
    ):
        print("Using OpenVINO ")
        if lcm_diffusion_setting.diffusion_task == DiffusionTask.text_to_image.value:
            return [
                self.pipeline.generate(
                    prompt=lcm_diffusion_setting.prompt,
                    neg_prompt=lcm_diffusion_setting.negative_prompt,
                    init_image=None,
                    strength=1.0,
                    num_inference_steps=lcm_diffusion_setting.inference_steps,
                )
            ]
        else:
            return [
                self.pipeline.generate(
                    prompt=lcm_diffusion_setting.prompt,
                    neg_prompt=lcm_diffusion_setting.negative_prompt,
                    init_image=lcm_diffusion_setting.init_image,
                    strength=lcm_diffusion_setting.strength,
                    num_inference_steps=lcm_diffusion_setting.inference_steps,
                )
            ]

    def _is_valid_mode(
        self,
        modes: List,
    ) -> bool:
        return modes.count(True) == 1 or modes.count(False) == 3

    def _validate_mode(
        self,
        modes: List,
    ) -> None:
        if not self._is_valid_mode(modes):
            raise ValueError("Invalid mode,delete configs/settings.yaml and retry!")

    def _is_sana_model(self) -> bool:
        return "sana" in self.ov_model_id.lower()

    def init(
        self,
        device: str = "cpu",
        lcm_diffusion_setting: LCMDiffusionSetting = LCMDiffusionSetting(),
    ) -> None:
        # Mode validation either LCM LoRA or OpenVINO or GGUF

        modes = [
            lcm_diffusion_setting.use_gguf_model,
            lcm_diffusion_setting.use_openvino,
            lcm_diffusion_setting.use_lcm_lora,
        ]
        self._validate_mode(modes)
        self.device = device
        self.use_openvino = lcm_diffusion_setting.use_openvino
        model_id = lcm_diffusion_setting.lcm_model_id
        use_local_model = lcm_diffusion_setting.use_offline_model
        use_tiny_auto_encoder = lcm_diffusion_setting.use_tiny_auto_encoder
        use_lora = lcm_diffusion_setting.use_lcm_lora
        lcm_lora: LCMLora = lcm_diffusion_setting.lcm_lora
        token_merging = lcm_diffusion_setting.token_merging
        self.ov_model_id = lcm_diffusion_setting.openvino_lcm_model_id

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
            or self.previous_ov_model_id != self.ov_model_id
            or self.previous_token_merging != token_merging
            or self.previous_safety_checker != lcm_diffusion_setting.use_safety_checker
            or self.previous_use_openvino != lcm_diffusion_setting.use_openvino
            or self.previous_use_gguf_model != lcm_diffusion_setting.use_gguf_model
            or self.previous_gguf_model != lcm_diffusion_setting.gguf_model
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
                    print(
                        f"***** Init Text to image (OpenVINO) - {self.ov_model_id} *****"
                    )
                    if "flux" in self.ov_model_id.lower() or self._is_sana_model():
                        if self._is_sana_model():
                            print("Loading OpenVINO SANA Sprint pipeline")
                        else:
                            print("Loading OpenVINO Flux pipeline")
                        self.pipeline = get_ov_diffusion_pipeline(self.ov_model_id)
                    elif self._is_hetero_pipeline():
                        self._load_ov_hetero_pipeline()
                    else:
                        self.pipeline = get_ov_text_to_image_pipeline(
                            self.ov_model_id,
                            use_local_model,
                        )
                elif (
                    lcm_diffusion_setting.diffusion_task
                    == DiffusionTask.image_to_image.value
                ):
                    if not self.pipeline and self._is_hetero_pipeline():
                        self._load_ov_hetero_pipeline()
                    else:
                        print(
                            f"***** Image to image (OpenVINO) - {self.ov_model_id} *****"
                        )
                        self.pipeline = get_ov_image_to_image_pipeline(
                            self.ov_model_id,
                            use_local_model,
                        )
            elif lcm_diffusion_setting.use_gguf_model:
                model = lcm_diffusion_setting.gguf_model.diffusion_path
                print(f"***** Init Text to image (GGUF) - {model} *****")
                # if self.pipeline:
                #     self.pipeline.terminate()
                #     del self.pipeline
                #     self.pipeline = None
                self._init_gguf_diffusion(lcm_diffusion_setting)
            else:
                reset_active_lora_weights()
                if self.pipeline or self.img_to_img_pipeline:
                    self.pipeline = self.txt2img_pipeline
                    self.txt2img_pipeline = None
                    self.img2img_pipeline = None
                    self.img_to_img_pipeline = None
                    self.controlnet_pipeline = None
                    self.controlnet_img2img_pipeline = None
                    del self.pipeline
                    self.pipeline = None
                    gc.collect()

                if use_lora:
                    print(
                        f"***** Init LCM-LoRA pipeline - {lcm_lora.base_model_id} *****"
                    )
                    self.pipeline = get_lcm_lora_pipeline(
                        lcm_lora.base_model_id,
                        lcm_lora.lcm_lora_id,
                        use_local_model,
                        torch_data_type=self.torch_data_type,
                    )

                else:
                    print(f"***** Init LCM Model pipeline - {model_id} *****")
                    self.pipeline = get_lcm_model_pipeline(
                        model_id,
                        use_local_model,
                        None,  # controlnet_args,
                    )

                # Prepare alternative generation pipelines using the txt2img pipeline from which all extra pipelines are derived
                self.txt2img_pipeline = self.pipeline
                self.img2img_pipeline = get_image_to_image_pipeline(self.pipeline)
                self.controlnet_pipeline = get_controlnet_pipeline(
                    self.pipeline, lcm_diffusion_setting, DiffusionTask.text_to_image
                )
                self.controlnet_img2img_pipeline = get_controlnet_pipeline(
                    self.pipeline, lcm_diffusion_setting, DiffusionTask.image_to_image
                )
                self.img_to_img_pipeline = self.img2img_pipeline

                if tomesd and token_merging > 0.001:
                    print(f"***** Token Merging: {token_merging} *****")
                    tomesd.apply_patch(self.pipeline, ratio=token_merging)
                    tomesd.apply_patch(self.img_to_img_pipeline, ratio=token_merging)

            if use_tiny_auto_encoder:
                if self.use_openvino and is_openvino_device():
                    if not self._is_sana_model():
                        print("Using Tiny AutoEncoder (OpenVINO)")
                        ov_load_tiny_autoencoder(
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

            if not self._is_hetero_pipeline():
                if (
                    lcm_diffusion_setting.diffusion_task
                    == DiffusionTask.image_to_image.value
                    and lcm_diffusion_setting.use_openvino
                ):
                    self.pipeline.scheduler = LCMScheduler.from_config(
                        self.pipeline.scheduler.config,
                    )
                else:
                    if not lcm_diffusion_setting.use_gguf_model:
                        self._update_lcm_scheduler_params()

            if use_lora:
                self._add_freeu()

            self.previous_model_id = model_id
            self.previous_ov_model_id = self.ov_model_id
            self.previous_use_tae_sd = use_tiny_auto_encoder
            self.previous_lcm_lora_base_id = lcm_lora.base_model_id
            self.previous_lcm_lora_id = lcm_lora.lcm_lora_id
            self.previous_use_lcm_lora = use_lora
            self.previous_token_merging = lcm_diffusion_setting.token_merging
            self.previous_safety_checker = lcm_diffusion_setting.use_safety_checker
            self.previous_use_openvino = lcm_diffusion_setting.use_openvino
            self.previous_task_type = lcm_diffusion_setting.diffusion_task
            self.previous_lora = lcm_diffusion_setting.lora.model_copy(deep=True)
            self.previous_use_gguf_model = lcm_diffusion_setting.use_gguf_model
            self.previous_gguf_model = lcm_diffusion_setting.gguf_model.model_copy(
                deep=True
            )
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
            elif not lcm_diffusion_setting.use_gguf_model:
                adapters = self.pipeline.get_active_adapters()
                print(f"Active adapters : {adapters}")

    def _get_timesteps(self):
        time_steps = self.pipeline.scheduler.config.get("timesteps")
        time_steps_value = [int(time_steps)] if time_steps else None
        return time_steps_value

    def _compile_ov_pipeline(
        self,
        lcm_diffusion_setting,
    ):
        self.pipeline.reshape(
            batch_size=-1,
            height=lcm_diffusion_setting.image_height,
            width=lcm_diffusion_setting.image_width,
            num_images_per_prompt=lcm_diffusion_setting.number_of_images,
        )
        self.pipeline.compile()

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

        self.pipeline = self.txt2img_pipeline
        self.img_to_img_pipeline = self.img2img_pipeline
        if (
            lcm_diffusion_setting.controlnet
            and lcm_diffusion_setting.controlnet.enabled
        ):
            if self.controlnet_pipeline != None:
                self.pipeline = self.controlnet_pipeline
            if self.controlnet_img2img_pipeline != None:
                self.img_to_img_pipeline = self.controlnet_img2img_pipeline
        pipeline_extra_args = {}

        if lcm_diffusion_setting.use_seed:
            cur_seed = lcm_diffusion_setting.seed
            # for multiple images with a fixed seed, use sequential seeds
            seeds = [
                (cur_seed + i) for i in range(lcm_diffusion_setting.number_of_images)
            ]
        else:
            seeds = [
                random.randint(0, 999999999)
                for i in range(lcm_diffusion_setting.number_of_images)
            ]

        if self.use_openvino:
            # no support for generators; try at least to ensure reproducible results for single images
            torch.manual_seed(seeds[0])
            if self._is_hetero_pipeline():
                torch.manual_seed(seeds[0])
                lcm_diffusion_setting.seed = seeds[0]
        else:
            pipeline_extra_args["generator"] = [
                torch.Generator(device=self.device).manual_seed(s) for s in seeds
            ]

        is_openvino_pipe = lcm_diffusion_setting.use_openvino and is_openvino_device()
        if is_openvino_pipe and not self._is_hetero_pipeline():
            print("Using OpenVINO")
            if self.is_openvino_init and self._is_sana_model():
                self._compile_ov_pipeline(lcm_diffusion_setting)

            if reshape and not self.is_openvino_init:
                print("Reshape and compile")
                self._compile_ov_pipeline(lcm_diffusion_setting)

            if self.is_openvino_init:
                self.is_openvino_init = False

        if is_openvino_pipe and self._is_hetero_pipeline():
            return self._generate_images_hetero_compute(lcm_diffusion_setting)
        elif lcm_diffusion_setting.use_gguf_model:
            return self._generate_images_gguf(lcm_diffusion_setting)

        if lcm_diffusion_setting.clip_skip > 1:
            # We follow the convention that "CLIP Skip == 2" means "skip
            # the last layer", so "CLIP Skip == 1" means "no skipping"
            pipeline_extra_args["clip_skip"] = lcm_diffusion_setting.clip_skip - 1

        self.pipeline.safety_checker = None
        if (
            lcm_diffusion_setting.diffusion_task == DiffusionTask.image_to_image.value
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
                if self._is_sana_model():
                    result_images = self.pipeline(
                        prompt=lcm_diffusion_setting.prompt,
                        num_inference_steps=lcm_diffusion_setting.inference_steps,
                        guidance_scale=guidance_scale,
                        width=lcm_diffusion_setting.image_width,
                        height=lcm_diffusion_setting.image_height,
                        num_images_per_prompt=lcm_diffusion_setting.number_of_images,
                    ).images
                else:
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
                    **pipeline_extra_args,
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
                    **pipeline_extra_args,
                    **controlnet_args,
                ).images

        for i, seed in enumerate(seeds):
            result_images[i].info["image_seed"] = seed

        return result_images

    def _init_gguf_diffusion(
        self,
        lcm_diffusion_setting: LCMDiffusionSetting,
    ):
        config = ModelConfig()
        config.model_path = lcm_diffusion_setting.gguf_model.diffusion_path
        config.diffusion_model_path = lcm_diffusion_setting.gguf_model.diffusion_path
        config.clip_l_path = lcm_diffusion_setting.gguf_model.clip_path
        config.t5xxl_path = lcm_diffusion_setting.gguf_model.t5xxl_path
        config.vae_path = lcm_diffusion_setting.gguf_model.vae_path
        config.n_threads = GGUF_THREADS
        print(f"GGUF Threads : {GGUF_THREADS} ")
        print("GGUF - Model config")
        pprint(lcm_diffusion_setting.gguf_model.model_dump())
        self.pipeline = GGUFDiffusion(
            get_app_path(),  # Place DLL in fastsdcpu folder
            config,
            True,
        )

    def _generate_images_gguf(
        self,
        lcm_diffusion_setting: LCMDiffusionSetting,
    ):
        if lcm_diffusion_setting.diffusion_task == DiffusionTask.text_to_image.value:
            t2iconfig = Txt2ImgConfig()
            t2iconfig.prompt = lcm_diffusion_setting.prompt
            t2iconfig.batch_count = lcm_diffusion_setting.number_of_images
            t2iconfig.cfg_scale = lcm_diffusion_setting.guidance_scale
            t2iconfig.height = lcm_diffusion_setting.image_height
            t2iconfig.width = lcm_diffusion_setting.image_width
            t2iconfig.sample_steps = lcm_diffusion_setting.inference_steps
            t2iconfig.sample_method = SampleMethod.EULER
            if lcm_diffusion_setting.use_seed:
                t2iconfig.seed = lcm_diffusion_setting.seed
            else:
                t2iconfig.seed = -1

            return self.pipeline.generate_text2mg(t2iconfig)
