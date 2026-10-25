"""OpenVINO pipeline for FLUX.3 Image Klein Base"""

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
import openvino as ov
from PIL import Image, ImageOps
from tqdm.auto import tqdm

CONFIG_NAME = "flux3_ov.json"
TEXT_ENCODER = "text_encoder/openvino_model.xml"
TRANSFORMER = "transformer/openvino_model.xml"
VAE_DECODER = "vae_decoder/openvino_model.xml"
VAE_ENCODER = "vae_encoder/openvino_model.xml"
TOKENIZER = "tokenizer"

DOWNSAMPLING_RATIO = 16  # 8x VAE + 2x2 patchify
LATENT_CHANNELS = 128
IMAGE_ROPE_SCALE = 0.5
AXES_DIM = (32, 32, 32, 32)
THETA = 10000
MAX_LENGTH = 4096
PAD_MULTIPLE = 80

# References, as in the reference pipeline except the single-reference limit: BFL fits it under 4 MP (2048x2048),
# 1 MP keeps editing on a CPU practical. The output takes the first reference's size unless one is given.
MAX_REFERENCES = 10
REFERENCE_MAX_PIXELS = (
    1024 * 1024
)  # single reference, padded canvas included (BFL: 4 MP)
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024  # each reference of a multi-reference edit
MIN_SIDE, MAX_SIDE = 64, 4096


def resolve_model_dir(
    model_id: str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    hf_cache_dir: str | Path | None = None,
) -> Path:
    """A local directory as is; otherwise a Hugging Face Hub repo id, downloaded once into the Hub cache and then
    reused. With ``local_files_only`` (or HF_HUB_OFFLINE=1) the cached snapshot is used without touching the network;
    without it, huggingface_hub also falls back to the cached snapshot when the Hub can't be reached."""
    if Path(model_id).is_dir():
        return Path(model_id)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(model_id),
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            cache_dir=hf_cache_dir,
        )
    )


def latent_ids(h: int, w: int) -> np.ndarray:
    """(h*w, 4) = (0, row, col, 0)."""
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    zeros = np.zeros_like(rows)
    return (
        np.stack([zeros, rows, cols, zeros], axis=-1)
        .reshape(h * w, 4)
        .astype(np.float32)
    )


def image_ids(h: int, w: int) -> np.ndarray:
    """Latent ids with the row/col axes scaled for the RoPE, for the generated image."""
    ids = latent_ids(h, w)
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids


def reference_ids(h: int, w: int, index: int) -> np.ndarray:
    """As image_ids; the l axis tells references apart (-1 for the first, -2 for the second, ...)."""
    ids = image_ids(h, w)
    ids[:, 3] = -(1 + index)
    return ids


def text_ids(length: int) -> np.ndarray:
    """(L, 4) = (0, 0, 0, position)."""
    ids = np.zeros((length, 4), dtype=np.float32)
    ids[:, 3] = np.arange(length)
    return ids


def rope_freqs(ids: np.ndarray) -> np.ndarray:
    """ids (L, 4) -> RoPE rotation matrices (1, 1, L, 64, 2, 2), float32."""
    parts = []
    for axis, dim in enumerate(AXES_DIM):
        scale = np.arange(0, dim, 2, dtype=np.float64) / dim
        angles = ids[:, axis].astype(np.float64)[:, None] * (1.0 / THETA**scale)[None]
        cos, sin = np.cos(angles), np.sin(angles)
        parts.append(
            np.stack([cos, -sin, sin, cos], axis=-1).reshape(*angles.shape, 2, 2)
        )
    return np.concatenate(parts, axis=-3)[None, None].astype(np.float32)


def timesteps(steps: int, shift: float) -> np.ndarray:
    """Flow-matching schedule from 1 to 0 with the time shift ``shift * t / (1 + (shift - 1) * t)``.
    shift=3 is ComfyUI's ModelSamplingFlux with max_shift = base_shift = ln 3 and the "simple" scheduler."""
    t = np.linspace(1.0, 0.0, steps + 1)
    return shift * t / (1.0 + (shift - 1.0) * t)


def _ceil16(v: int) -> int:
    return -(-v // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO


def reference_canvas(
    width: int, height: int, max_pixels: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """(fitted size, canvas) of a single reference: the canvas (the fitted size rounded up to 16) at most
    ``max_pixels``, sides 64-4096."""
    scale = min(
        1.0, math.sqrt(max_pixels / (width * height)), MAX_SIDE / max(width, height)
    )
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, _ceil16(v))) for v in fit)
        if canvas[0] * canvas[1] <= max_pixels:
            return fit, canvas
        scale *= 0.995


def prepare_reference(
    image: Image.Image, max_pixels: int = REFERENCE_MAX_PIXELS
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """A single reference: RGB, fitted by reference_canvas and reflect-padded to its canvas, centred. Returns the
    image and, for a padded one, the (left, top, right, bottom) box of the fitted image inside it."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height, max_pixels)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image, None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    mode = (
        "reflect" if min(image.size) > DOWNSAMPLING_RATIO else "symmetric"
    )  # reflect needs a pad shorter than the side
    return Image.fromarray(np.pad(np.asarray(image), padding, mode=mode)), (
        left,
        top,
        left + image.width,
        top + image.height,
    )


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    """Each reference of a multi-reference edit: RGB, fitted under 1 MP (sides floored), centre-cropped to the
    16-px grid."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    w, h = (
        width // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO,
        height // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO,
    )
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))


class FlowMatchEulerScheduler:
    """Euler steps of the flow-matching ODE from t=1 (noise) to 0 on ``timesteps(steps, shift)``, in numpy, with the
    method names of diffusers' FlowMatchEulerDiscreteScheduler (fixed shift, no dynamic shifting). Unlike diffusers,
    ``timesteps`` are t in [0, 1] as the transformer takes them, not 1000 t."""

    order = 1

    def __init__(self, shift: float = 3.0):
        self.config = SimpleNamespace(shift=shift)
        self.sigmas = self.timesteps = np.zeros(0, np.float32)
        self._step_index = 0

    def set_timesteps(
        self, num_inference_steps: int, shift: float | None = None
    ) -> None:
        self.sigmas = timesteps(num_inference_steps, shift or self.config.shift)
        self.timesteps = self.sigmas[:-1].astype(np.float32)
        self._step_index = 0

    def step(
        self, model_output: np.ndarray, timestep: float, sample: np.ndarray
    ) -> np.ndarray:
        """One Euler step: sample + (t_next - t) * velocity. Steps must follow ``timesteps`` in order."""
        dt = self.sigmas[self._step_index + 1] - self.sigmas[self._step_index]
        self._step_index += 1
        return sample + np.float32(dt) * model_output


@dataclass
class Flux3KleinPipelineOutput:
    """As diffusers' pipeline outputs: ``images`` is a list of PIL images, or an array for output_type "np"
    (N, H, W, 3) in [0, 1] or "latent" (N, 128, H/16, W/16)."""

    images: list[Image.Image] | np.ndarray


def seeds_for(count: int, seed: int | None, generator) -> list[int]:
    """One seed per image: ``seed + i``; else from ``generator`` (an int seed, a torch.Generator, numpy
    Generator/RandomState, or a list of them, one per image); else drawn from numpy's global state, so
    np.random.seed() repeats a run."""
    if seed is not None:
        return [(seed + i) % 2**32 for i in range(count)]
    if isinstance(generator, (int, np.integer)):
        return [(int(generator) + i) % 2**32 for i in range(count)]
    if isinstance(generator, (list, tuple)):
        if len(generator) != count:
            raise ValueError(f"{len(generator)} generators for {count} images")
        return [seeds_for(1, None, g)[0] for g in generator]
    if hasattr(
        generator, "initial_seed"
    ):  # torch.Generator: its noise can't be reproduced without torch
        return [(generator.initial_seed() + i) % 2**32 for i in range(count)]
    if isinstance(generator, np.random.Generator):
        return [int(s) for s in generator.integers(2**32, size=count)]
    source = generator if isinstance(generator, np.random.RandomState) else np.random
    return [int(s) for s in source.randint(2**32, size=count, dtype=np.int64)]


class OVFlux3KleinPipeline:
    """FLUX.3 Image Klein text-to-image and reference editing on OpenVINO, called like a diffusers / optimum-intel
    OVDiffusionPipeline.

    >>> pipe = OVFlux3KleinPipeline.from_pretrained("models/flux3-klein-4step-ov-int4")  # or a Hub repo id
    >>> image = pipe("a red fox in fresh snow, golden hour", width=1024, height=1024, seed=42).images[0]
    >>> night = pipe("the same scene at night under a full moon", image=image, seed=42).images[0]
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "CPU",
        cache_dir: str | Path | None = None,
        ov_config: dict | None = None,
        compile: bool = True,
    ):
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / CONFIG_NAME).read_text())
        self._device = device.upper()
        self.core = ov.Core()
        if cache_dir is not None:
            self.core.set_property({"CACHE_DIR": str(cache_dir)})
        self.ov_config = {"PERFORMANCE_HINT": "LATENCY", **(ov_config or {})}
        self.text_encoder = self.transformer = self.vae_decoder = self.vae_encoder = (
            None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / TOKENIZER)
        self.tokenizer.padding_side = "right"
        self.scheduler = FlowMatchEulerScheduler(self.config["shift"])
        self.timings: dict[str, list[float]] = {"steps": []}
        self._progress_bar_config: dict = {}
        if compile:
            self.compile()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        hf_cache_dir: str | Path | None = None,
        **kwargs,
    ) -> "OVFlux3KleinPipeline":
        """``model_id`` is a model directory or a Hub repo id (see ``resolve_model_dir``). ``kwargs`` go to the
        constructor: ``device``, ``ov_config``, ``compile`` as in optimum-intel, and ``cache_dir``, which here is the
        OpenVINO compiled-model cache (optimum's ``ov_config={"CACHE_DIR": ...}`` works too); ``hf_cache_dir`` is the
        Hub download cache."""
        return cls(
            resolve_model_dir(
                model_id, revision, token, local_files_only, hf_cache_dir
            ),
            **kwargs,
        )

    @property
    def device(self) -> str:
        return self._device

    @property
    def num_timesteps(self) -> int:
        """Denoising steps of the current (or last) call."""
        return len(self.scheduler.timesteps)

    def compile(self) -> None:
        """Compile the models for ``device``; done on first use when loaded with ``compile=False`` or after ``to``."""
        if self.transformer is not None:
            return
        # GPU runs in FP16 and the transformer's activations overflow (garbage images); scaling them down keeps FP16
        # speed, while INFERENCE_PRECISION_HINT=f32 also works but is ~14x slower per step on an iGPU
        scaled = (
            {"ACTIVATIONS_SCALE_FACTOR": 8.0, **self.ov_config}
            if self._device != "CPU"
            else self.ov_config
        )
        compile = lambda name, config=self.ov_config: self.core.compile_model(
            self.model_dir / name, self._device, config
        ).create_infer_request()
        self.text_encoder = compile(TEXT_ENCODER, scaled)
        self.transformer = compile(TRANSFORMER, scaled)
        self.vae_decoder = compile(VAE_DECODER)
        self.vae_encoder = (
            compile(VAE_ENCODER) if (self.model_dir / VAE_ENCODER).exists() else None
        )

    def clear_requests(self) -> None:
        self.text_encoder = self.transformer = self.vae_decoder = self.vae_encoder = (
            None
        )

    def set_progress_bar_config(self, **kwargs) -> None:
        """tqdm options for the per-image denoising bar, as in diffusers (e.g. ``disable=True``, ``leave=False``)."""
        self._progress_bar_config = kwargs

    def progress_bar(self, iterable=None, total: int | None = None) -> tqdm:
        return tqdm(iterable, total=total, **self._progress_bar_config)

    def reshape(
        self,
        batch_size: int = -1,
        height: int = -1,
        width: int = -1,
        num_images_per_prompt: int = -1,
        **kwargs,
    ) -> "OVFlux3KleinPipeline":
        """As optimum-intel's reshape, but the models stay dynamic: they take any size on the 16-px grid, so this
        only checks the size and keeps the compiled models (no recompile when an app changes the size)."""
        for name, value in (("height", height), ("width", width)):
            if value != -1 and value % DOWNSAMPLING_RATIO:
                raise ValueError(
                    f"{name} must be a multiple of {DOWNSAMPLING_RATIO}, not {value}"
                )
        return self

    def to(self, device: str) -> "OVFlux3KleinPipeline":
        """Move to another OpenVINO device ("CPU", "GPU", ...); the models recompile on the next call."""
        self._device = device.upper()
        self.clear_requests()
        return self

    def encode_prompt(self, prompt: str) -> np.ndarray:
        """prompt -> text tokens (L, 20480); L is the chat-templated prompt padded to a multiple of 80."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        real = len(
            self.tokenizer(text, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        )
        length = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=length,
        )
        result = self.text_encoder.infer(
            [
                inputs["input_ids"].astype(np.int64),
                inputs["attention_mask"].astype(np.int64),
            ]
        )
        return result[0].copy()

    def encode_references(
        self, references: Sequence[Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepared references (RGB, sides multiples of 16) -> latent tokens (M, 128) and ids (M, 4); M = 0 for none."""
        if not references:
            return np.zeros((0, LATENT_CHANNELS), np.float32), np.zeros(
                (0, 4), np.float32
            )
        if self.vae_encoder is None:
            raise RuntimeError(
                f"{self.model_dir} has no {VAE_ENCODER}; re-run flux3-ov-export to edit images"
            )
        tokens, ids = [], []
        for index, reference in enumerate(references):
            pixels = (
                np.asarray(reference, dtype=np.float32).transpose(2, 0, 1)[None] / 127.5
                - 1.0
            )
            tokens.append(self.vae_encoder.infer([pixels])[0].copy())
            ids.append(
                reference_ids(
                    reference.height // DOWNSAMPLING_RATIO,
                    reference.width // DOWNSAMPLING_RATIO,
                    index,
                )
            )
        return np.concatenate(tokens), np.concatenate(ids)

    def __call__(
        self,
        prompt: str | Sequence[str],
        image: Image.Image | Sequence[Image.Image] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | Sequence[str] | None = None,
        num_images_per_prompt: int = 1,
        generator=None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[["OVFlux3KleinPipeline", int, float, dict], dict]
        | None = None,
        callback_on_step_end_tensor_inputs: Sequence[str] = ("latents",),
        shift: float | None = None,
        seed: int | None = None,
        reference_max_pixels: int = REFERENCE_MAX_PIXELS,
    ) -> Flux3KleinPipelineOutput | tuple:
        """Generate ``num_images_per_prompt`` images per prompt; with ``image`` (one, or a list of up to 10
        references), edit or compose them as the prompt says. Returns ``.images`` as diffusers pipelines do, or
        ``(images,)`` with ``return_dict=False``.

        The size defaults to 1024x1024, or to the first prepared reference's size. A single reference is fitted
        under ``reference_max_pixels`` and padded to the 16-px grid; the padding is cropped off the output when the
        size is left to default. Seeds: see ``seeds_for``; ``last_seed`` is the first image's.
        ``callback_on_step_end(pipe, step, t, {"latents": x})`` runs after every denoising step and may return
        replacement latents; ``timings["steps"]`` holds the seconds of each step so far.

        ``guidance_scale`` and ``negative_prompt`` are accepted for diffusers compatibility and ignored: the
        few-step model runs without CFG.
        """
        if guidance_scale is not None and guidance_scale > 1:
            warnings.warn(
                f"guidance_scale={guidance_scale} is ignored: the few-step model runs without CFG"
            )
        if negative_prompt:
            warnings.warn(
                "negative_prompt is ignored: it needs CFG, which the few-step model runs without"
            )
        if output_type not in ("pil", "np", "latent"):
            raise ValueError(
                f"output_type must be 'pil', 'np' or 'latent', not {output_type!r}"
            )
        if set(callback_on_step_end_tensor_inputs) - {"latents"}:
            raise ValueError(
                "callback_on_step_end_tensor_inputs supports only 'latents'"
            )
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        images = [image] if isinstance(image, Image.Image) else list(image or ())
        if len(images) > MAX_REFERENCES:
            raise ValueError(f"at most {MAX_REFERENCES} reference images")
        if len(images) > 1:
            references, crop = (
                [prepare_multi_reference(image) for image in images],
                None,
            )
        else:
            prepared = [
                prepare_reference(image, reference_max_pixels) for image in images
            ]
            references = [image for image, _ in prepared]
            crop = prepared[0][1] if prepared and not (width or height) else None
        default_width, default_height = (
            references[0].size if references else (1024, 1024)
        )
        width, height = width or default_width, height or default_height
        if width % DOWNSAMPLING_RATIO or height % DOWNSAMPLING_RATIO:
            raise ValueError(
                f"width and height must be multiples of {DOWNSAMPLING_RATIO}"
            )
        steps = num_inference_steps or self.config["steps"]
        shift = shift or self.config["shift"]
        seeds = seeds_for(len(prompts) * num_images_per_prompt, seed, generator)
        self.last_seed = seeds[0]
        self.timings = {"steps": []}
        self.compile()

        cond, cond_ids = self.encode_references(references)
        h, w = height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO
        image_freqs, cond_freqs = rope_freqs(image_ids(h, w)), rope_freqs(cond_ids)
        outputs = []
        for index, seed in enumerate(seeds):
            if index % num_images_per_prompt == 0:
                text = self.encode_prompt(prompts[index // num_images_per_prompt])
                text_freqs = rope_freqs(text_ids(text.shape[0]))
            x = np.random.default_rng(seed).standard_normal(
                (h * w, LATENT_CHANNELS), dtype=np.float32
            )
            self.scheduler.set_timesteps(steps, shift)
            for step, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                start = time.perf_counter()
                velocity = self.transformer.infer(
                    [
                        x,
                        image_freqs,
                        cond,
                        cond_freqs,
                        text,
                        text_freqs,
                        np.array([t], dtype=np.float32),
                    ]
                )[0]
                x = self.scheduler.step(velocity, t, x)
                self.timings["steps"].append(time.perf_counter() - start)
                if callback_on_step_end is not None:
                    x = (
                        callback_on_step_end(self, step, float(t), {"latents": x}) or {}
                    ).get("latents", x)

            latent = np.ascontiguousarray(
                x.reshape(h, w, LATENT_CHANNELS).transpose(2, 0, 1)[None]
            )
            if output_type == "latent":
                outputs.append(latent[0])
                continue
            pixels = self.vae_decoder.infer([latent])[0][0]
            array = (
                ((np.clip(pixels, -1.0, 1.0) + 1.0) * 127.5)
                .astype(np.uint8)
                .transpose(1, 2, 0)
            )
            result = Image.fromarray(array)
            outputs.append(result.crop(crop) if crop else result)

        if output_type == "np":
            outputs = np.stack(
                [np.asarray(result, dtype=np.float32) / 255.0 for result in outputs]
            )
        elif output_type == "latent":
            outputs = np.stack(outputs)
        return Flux3KleinPipelineOutput(images=outputs) if return_dict else (outputs,)
