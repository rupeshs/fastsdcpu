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
from constants import (
    DEVICE,
    LCM_DEFAULT_MODEL,
    TAESD_MODEL,
    TAESDXL_MODEL,
    TAESD_MODEL_OPENVINO,
)
from huggingface_hub import model_info
from backend.models.lcmdiffusion_setting import LCMLora
from backend.device import is_openvino_device

if is_openvino_device():
    from huggingface_hub import snapshot_download
    from optimum.intel.openvino.modeling_diffusion import OVModelVaeDecoder, OVBaseModel

    # from optimum.intel.openvino.modeling_diffusion import OVStableDiffusionPipeline
    from backend.lcmdiffusion.pipelines.openvino.lcm_ov_pipeline import (
        OVStableDiffusionPipeline,
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
        self.device = ""
        self.previous_model_id = None
        self.previous_use_tae_sd = False
        self.previous_use_lcm_lora = False
        self.torch_data_type = (
            torch.float32 if is_openvino_device() or DEVICE == "mps" else torch.float16
        )
        print(f"Torch datatype : {self.torch_data_type}")

    def _get_lcm_pipeline(
        self,
        lcm_model_id: str,
        base_model_id: str,
        use_local_model: bool,
    ):
        pipeline = None
        unet = UNet2DConditionModel.from_pretrained(
            lcm_model_id,
            torch_dtype=torch.float32,
            local_files_only=use_local_model
            # resume_download=True,
        )
        pipeline = DiffusionPipeline.from_pretrained(
            base_model_id,
            unet=unet,
            torch_dtype=torch.float32,
            local_files_only=use_local_model
            # resume_download=True,
        )
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        return pipeline

    def get_tiny_decoder_vae_model(self) -> str:
        pipeline_class = self.pipeline.__class__.__name__
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

    def _get_lcm_model_pipeline(
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
                use_local_model,
            )

        elif model_id == "latent-consistency/lcm-ssd-1b":
            pipeline = self._get_lcm_pipeline(
                model_id,
                "segmind/SSD-1B",
                use_local_model,
            )
        return pipeline

    def _get_lcm_lora_pipeline(
        self,
        base_model_id: str,
        lcm_lora_id: str,
        use_local_model: bool,
    ):
        pipeline = DiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=self.torch_data_type,
            local_files_only=use_local_model,
        )
        pipeline.load_lora_weights(
            lcm_lora_id,
            local_files_only=use_local_model,
        )

        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

        pipeline.fuse_lora()
        pipeline.unet.to(memory_format=torch.channels_last)
        return pipeline

    def _pipeline_to_device(self):
        print(f"Pipeline device : {DEVICE}")
        print(f"Pipeline dtype : {self.torch_data_type}")
        self.pipeline.to(
            torch_device=DEVICE,
            torch_dtype=self.torch_data_type,
        )

    def _add_freeu(self):
        pipeline_class = self.pipeline.__class__.__name__
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

    def init(
        self,
        model_id: str,
        use_openvino: bool = False,
        device: str = "cpu",
        use_local_model: bool = False,
        use_tiny_auto_encoder: bool = False,
        use_lora: bool = False,
        lcm_lora: LCMLora = LCMLora(),
    ) -> None:
        self.device = device
        self.use_openvino = use_openvino
        if (
            self.pipeline is None
            or self.previous_model_id != model_id
            or self.previous_use_tae_sd != use_tiny_auto_encoder
            or self.previous_lcm_lora_base_id != lcm_lora.base_model_id
            or self.previous_lcm_lora_id != lcm_lora.lcm_lora_id
            or self.previous_use_lcm_lora != use_lora
        ):
            if self.use_openvino and is_openvino_device():
                if self.pipeline:
                    del self.pipeline
                    self.pipeline = None

                self.pipeline = OVStableDiffusionPipeline.from_pretrained(
                    model_id,
                    local_files_only=use_local_model,
                    ov_config={"CACHE_DIR": ""},
                    device=DEVICE.upper(),
                )

                if use_tiny_auto_encoder:
                    print("Using Tiny Auto Encoder (OpenVINO)")
                    taesd_dir = snapshot_download(
                        repo_id=self.get_tiny_decoder_vae_model(),
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
                    self.pipeline = None

                if use_lora:
                    print("Init LCM-LoRA pipeline")
                    self.pipeline = self._get_lcm_lora_pipeline(
                        lcm_lora.base_model_id,
                        lcm_lora.lcm_lora_id,
                        use_local_model,
                    )
                else:
                    print("Init LCM Model pipeline")
                    self.pipeline = self._get_lcm_model_pipeline(
                        model_id,
                        use_local_model,
                    )

                if use_tiny_auto_encoder:
                    vae_model = self.get_tiny_decoder_vae_model()
                    print(f"Using Tiny Auto Encoder {vae_model}")
                    self.pipeline.vae = AutoencoderTiny.from_pretrained(
                        vae_model,
                        torch_dtype=torch.float32,
                        local_files_only=use_local_model,
                    )

                self._pipeline_to_device()

            self.previous_model_id = model_id
            self.previous_use_tae_sd = use_tiny_auto_encoder
            self.previous_lcm_lora_base_id = lcm_lora.base_model_id
            self.previous_lcm_lora_id = lcm_lora.lcm_lora_id
            self.previous_use_lcm_lora = use_lora
            print(f"Model :{model_id}")
            print(f"Pipeline : {self.pipeline}")
            self.pipeline.scheduler = LCMScheduler.from_config(
                self.pipeline.scheduler.config,
                beta_start=0.001,
                beta_end=0.01,
            )
            if use_lora:
                self._add_freeu()

    def generate(
        self,
        lcm_diffusion_setting: LCMDiffusionSetting,
        reshape: bool = False,
    ) -> Any:
        guidance_scale = lcm_diffusion_setting.guidance_scale
        if lcm_diffusion_setting.use_seed:
            cur_seed = lcm_diffusion_setting.seed
            if self.use_openvino:
                np.random.seed(cur_seed)
            else:
                torch.manual_seed(cur_seed)

        if lcm_diffusion_setting.use_openvino and is_openvino_device():
            print("Using OpenVINO")
            if reshape:
                print("Reshape and compile")
                self.pipeline.reshape(
                    batch_size=-1,
                    height=lcm_diffusion_setting.image_height,
                    width=lcm_diffusion_setting.image_width,
                    num_images_per_prompt=lcm_diffusion_setting.number_of_images,
                )
                self.pipeline.compile()

        if not lcm_diffusion_setting.use_safety_checker:
            self.pipeline.safety_checker = None

        if (
            not lcm_diffusion_setting.use_lcm_lora
            and not lcm_diffusion_setting.use_openvino
            and lcm_diffusion_setting.guidance_scale != 1.0
        ):
            print("Not using LCM-LoRA so setting guidance_scale 1.0")
            guidance_scale = 1.0

        if lcm_diffusion_setting.use_openvino:
            result_images = self.pipeline(
                prompt=lcm_diffusion_setting.prompt,
                negative_prompt=lcm_diffusion_setting.negative_prompt,
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

        return result_images
