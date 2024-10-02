"""
Wrapper class to call the stablediffusion.cpp shared library for GGUF support
"""

import ctypes
import platform
from ctypes import (
    POINTER,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_int64,
    c_void_p,
)
from dataclasses import dataclass
from os import path
from typing import List, Any

import numpy as np
from PIL import Image

from backend.gguf.sdcpp_types import (
    RngType,
    SampleMethod,
    Schedule,
    SDCPPLogLevel,
    SDImage,
    SdType,
)


@dataclass
class ModelConfig:
    model_path: str = ""
    clip_l_path: str = ""
    t5xxl_path: str = ""
    diffusion_model_path: str = ""
    vae_path: str = ""
    taesd_path: str = ""
    control_net_path: str = ""
    lora_model_dir: str = ""
    embed_dir: str = ""
    stacked_id_embed_dir: str = ""
    vae_decode_only: bool = True
    vae_tiling: bool = False
    free_params_immediately: bool = False
    n_threads: int = 4
    wtype: SdType = SdType.SD_TYPE_Q4_0
    rng_type: RngType = RngType.CUDA_RNG
    schedule: Schedule = Schedule.DEFAULT
    keep_clip_on_cpu: bool = False
    keep_control_net_cpu: bool = False
    keep_vae_on_cpu: bool = False


@dataclass
class Txt2ImgConfig:
    prompt: str = "a man wearing sun glasses, highly detailed"
    negative_prompt: str = ""
    clip_skip: int = -1
    cfg_scale: float = 2.0
    guidance: float = 3.5
    width: int = 512
    height: int = 512
    sample_method: SampleMethod = SampleMethod.EULER_A
    sample_steps: int = 1
    seed: int = -1
    batch_count: int = 2
    control_cond: Image = None
    control_strength: float = 0.90
    style_strength: float = 0.5
    normalize_input: bool = False
    input_id_images_path: bytes = b""


class GGUFDiffusion:
    """GGUF Diffusion
    To support GGUF diffusion model based on stablediffusion.cpp
    https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    Implmented based on stablediffusion.h
    """

    def __init__(
        self,
        libpath: str,
        config: ModelConfig,
        logging_enabled: bool = False,
    ):
        sdcpp_shared_lib_path = self._get_sdcpp_shared_lib_path(libpath)
        try:
            self.libsdcpp = ctypes.CDLL(sdcpp_shared_lib_path)
        except OSError as e:
            print(f"Failed to load library {sdcpp_shared_lib_path}")
            raise ValueError(f"Error: {e}")

        if not config.clip_l_path or not path.exists(config.clip_l_path):
            raise ValueError(
                "CLIP model file not found,please check readme.md for GGUF model usage"
            )

        if not config.t5xxl_path or not path.exists(config.t5xxl_path):
            raise ValueError(
                "T5XXL model file not found,please check readme.md for GGUF model usage"
            )

        if not config.diffusion_model_path or not path.exists(
            config.diffusion_model_path
        ):
            raise ValueError(
                "Diffusion model file not found,please check readme.md for GGUF model usage"
            )

        if not config.vae_path or not path.exists(config.vae_path):
            raise ValueError(
                "VAE model file not found,please check readme.md for GGUF model usage"
            )

        self.model_config = config

        self.libsdcpp.new_sd_ctx.argtypes = [
            c_char_p,  # const char* model_path
            c_char_p,  # const char* clip_l_path
            c_char_p,  # const char* t5xxl_path
            c_char_p,  # const char* diffusion_model_path
            c_char_p,  # const char* vae_path
            c_char_p,  # const char* taesd_path
            c_char_p,  # const char* control_net_path_c_str
            c_char_p,  # const char* lora_model_dir
            c_char_p,  # const char* embed_dir_c_str
            c_char_p,  # const char* stacked_id_embed_dir_c_str
            c_bool,  # bool vae_decode_only
            c_bool,  # bool vae_tiling
            c_bool,  # bool free_params_immediately
            c_int,  # int n_threads
            SdType,  # enum sd_type_t wtype
            RngType,  # enum rng_type_t rng_type
            Schedule,  # enum schedule_t s
            c_bool,  # bool keep_clip_on_cpu
            c_bool,  # bool keep_control_net_cpu
            c_bool,  # bool keep_vae_on_cpu
        ]

        self.libsdcpp.new_sd_ctx.restype = POINTER(c_void_p)

        self.sd_ctx = self.libsdcpp.new_sd_ctx(
            self._str_to_bytes(self.model_config.model_path),
            self._str_to_bytes(self.model_config.clip_l_path),
            self._str_to_bytes(self.model_config.t5xxl_path),
            self._str_to_bytes(self.model_config.diffusion_model_path),
            self._str_to_bytes(self.model_config.vae_path),
            self._str_to_bytes(self.model_config.taesd_path),
            self._str_to_bytes(self.model_config.control_net_path),
            self._str_to_bytes(self.model_config.lora_model_dir),
            self._str_to_bytes(self.model_config.embed_dir),
            self._str_to_bytes(self.model_config.stacked_id_embed_dir),
            self.model_config.vae_decode_only,
            self.model_config.vae_tiling,
            self.model_config.free_params_immediately,
            self.model_config.n_threads,
            self.model_config.wtype,
            self.model_config.rng_type,
            self.model_config.schedule,
            self.model_config.keep_clip_on_cpu,
            self.model_config.keep_control_net_cpu,
            self.model_config.keep_vae_on_cpu,
        )

        if logging_enabled:
            self._set_logcallback()

    def _set_logcallback(self):
        print("Setting logging callback")
        # Define function callback
        SdLogCallbackType = ctypes.CFUNCTYPE(
            None,
            SDCPPLogLevel,
            ctypes.c_char_p,
            ctypes.c_void_p,
        )

        self.libsdcpp.sd_set_log_callback.argtypes = [
            SdLogCallbackType,
            ctypes.c_void_p,
        ]
        self.libsdcpp.sd_set_log_callback.restype = None
        # Convert the Python callback to a C func pointer
        self.c_log_callback = SdLogCallbackType(
            self.log_callback
        )  # prevent GC,keep callback as member variable
        self.libsdcpp.sd_set_log_callback(self.c_log_callback, None)

    def _get_sdcpp_shared_lib_path(
        self,
        root_path: str,
    ) -> str:
        system_name = platform.system()
        print(f"GGUF Diffusion on {system_name}")
        lib_name = "stable-diffusion.dll"
        sdcpp_lib_path = ""

        if system_name == "Windows":
            sdcpp_lib_path = path.join(root_path, lib_name)
        elif system_name == "Linux":
            lib_name = "libstable-diffusion.so"
            sdcpp_lib_path = path.join(root_path, lib_name)
        elif system_name == "Darwin":
            lib_name = "libstable-diffusion.dylib"
            sdcpp_lib_path = path.join(root_path, lib_name)
        else:
            print("Unknown platform.")

        return sdcpp_lib_path

    @staticmethod
    def log_callback(
        level,
        text,
        data,
    ):
        print(f"{text.decode('utf-8')}", end="")

    def _str_to_bytes(self, in_str: str, encoding: str = "utf-8") -> bytes:
        if in_str:
            return in_str.encode(encoding)
        else:
            return b""

    def generate_text2mg(self, txt2img_cfg: Txt2ImgConfig) -> List[Any]:
        self.libsdcpp.txt2img.restype = POINTER(SDImage)
        self.libsdcpp.txt2img.argtypes = [
            c_void_p,  # sd_ctx_t* sd_ctx (pointer to context object)
            c_char_p,  # const char* prompt
            c_char_p,  # const char* negative_prompt
            c_int,  # int clip_skip
            c_float,  # float cfg_scale
            c_float,  # float guidance
            c_int,  # int width
            c_int,  # int height
            SampleMethod,  # enum sample_method_t sample_method
            c_int,  # int sample_steps
            c_int64,  # int64_t seed
            c_int,  # int batch_count
            POINTER(SDImage),  # const sd_image_t* control_cond (pointer to SDImage)
            c_float,  # float control_strength
            c_float,  # float style_strength
            c_bool,  # bool normalize_input
            c_char_p,  # const char* input_id_images_path
        ]

        image_buffer = self.libsdcpp.txt2img(
            self.sd_ctx,
            self._str_to_bytes(txt2img_cfg.prompt),
            self._str_to_bytes(txt2img_cfg.negative_prompt),
            txt2img_cfg.clip_skip,
            txt2img_cfg.cfg_scale,
            txt2img_cfg.guidance,
            txt2img_cfg.width,
            txt2img_cfg.height,
            txt2img_cfg.sample_method,
            txt2img_cfg.sample_steps,
            txt2img_cfg.seed,
            txt2img_cfg.batch_count,
            txt2img_cfg.control_cond,
            txt2img_cfg.control_strength,
            txt2img_cfg.style_strength,
            txt2img_cfg.normalize_input,
            txt2img_cfg.input_id_images_path,
        )

        images = self._get_sd_images_from_buffer(
            image_buffer,
            txt2img_cfg.batch_count,
        )

        return images

    def _get_sd_images_from_buffer(
        self,
        image_buffer: Any,
        batch_count: int,
    ) -> List[Any]:
        images = []
        if image_buffer:
            for i in range(batch_count):
                image = image_buffer[i]
                print(
                    f"Generated image: {image.width}x{image.height} with {image.channel} channels"
                )

                width = image.width
                height = image.height
                channels = image.channel
                pixel_data = np.ctypeslib.as_array(
                    image.data, shape=(height, width, channels)
                )

                if channels == 1:
                    pil_image = Image.fromarray(pixel_data.squeeze(), mode="L")
                elif channels == 3:
                    pil_image = Image.fromarray(pixel_data, mode="RGB")
                elif channels == 4:
                    pil_image = Image.fromarray(pixel_data, mode="RGBA")
                else:
                    raise ValueError(f"Unsupported number of channels: {channels}")

                images.append(pil_image)
        return images

    def terminate(self):
        if self.libsdcpp:
            if self.sd_ctx:
                self.libsdcpp.free_sd_ctx.argtypes = [c_void_p]
                self.libsdcpp.free_sd_ctx.restype = None
                self.libsdcpp.free_sd_ctx(self.sd_ctx)
                del self.sd_ctx
                self.sd_ctx = None
                del self.libsdcpp
                self.libsdcpp = None
