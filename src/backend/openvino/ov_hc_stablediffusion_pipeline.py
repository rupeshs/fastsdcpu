"""This is an experimental pipeline used to test AI PC NPU and GPU"""

from pathlib import Path

from diffusers import EulerDiscreteScheduler,LCMScheduler
from huggingface_hub import snapshot_download
from PIL import Image
from backend.openvino.stable_diffusion_engine import (
    StableDiffusionEngineAdvanced,
    LatentConsistencyEngine
)


class OvHcStableDiffusion:
    "OpenVINO Heterogeneous compute Stablediffusion"

    def __init__(
        self,
        model_path,
        device: list = ["GPU", "NPU", "GPU", "GPU"],
    ):
        model_dir = Path(snapshot_download(model_path))
        self.scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )
        self.ov_sd_pipleline = StableDiffusionEngineAdvanced(
            model=model_dir,
            device=device,
        )

    def generate(
        self,
        prompt: str,
        neg_prompt: str,
        init_image: Image = None,
        strength: float = 1.0,
    ):
        image = self.ov_sd_pipleline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            init_image=init_image,
            strength=strength,
            num_inference_steps=25,
            scheduler=self.scheduler,
        )
        image_rgb = image[..., ::-1]
        return Image.fromarray(image_rgb)


class OvHcLatentConsistency:
    """
    OpenVINO Heterogeneous compute Latent consistency models
    For the current Intel Cor Ultra, the Text Encoder and Unet can run on NPU

    """

    def __init__(
        self,
        model_path,
        device: list = ["NPU", "NPU", "GPU"],
    ):
        
        model_dir = Path(snapshot_download(model_path))
      
        self.scheduler = LCMScheduler(
                beta_start=0.001,
                beta_end=0.01,
            )
        self.ov_sd_pipleline = LatentConsistencyEngine(
            model=model_dir,
            device=device,
        )

    def generate(
        self,
        prompt: str,
        neg_prompt: str,
        init_image: Image = None,
         num_inference_steps=4,
        strength: float = 1.0,
    ):
        image = self.ov_sd_pipleline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            scheduler=self.scheduler,
            seed=None,
        )
        
        return image
