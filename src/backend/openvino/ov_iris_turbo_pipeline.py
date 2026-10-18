"""OpenVINO pipeline for Iris-3B Turbo (pixel-space DiT: 4 steps, no CFG, no VAE)."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import openvino as ov
from PIL import Image
from tokenizers import Tokenizer

Seed = Union[int, np.random.Generator]

# INT8 export of Iris-3B Turbo (1024x1024) on the Hugging Face Hub
DEFAULT_MODEL_ID = "rupeshs/iris-3b-int8-ov"
# everything the pipeline reads from a model folder; nothing else is downloaded
MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "text_encoder.xml",
    "text_encoder.bin",
    "transformer.xml",
    "transformer.bin",
)


def timestep_features(t: float, freq_dim: int, max_period: float) -> np.ndarray:
    """Sinusoidal timestep features, [1, freq_dim].

    Computed here rather than in the graph so FP16 weights or FP16 execution
    cannot round the frequencies that multiply ``t``.
    """
    n = freq_dim // 2
    phase = t * np.exp(-np.log(max_period) * np.arange(n, dtype=np.float64) / n)
    return np.concatenate([np.cos(phase), np.sin(phase)])[None].astype(np.float32)


@dataclass
class OVIrisTurboPipelineOutput:
    images: Union[List[Image.Image], np.ndarray]


class OVIrisTurboPipeline:
    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str = "CPU",
        text_encoder_device: Optional[str] = None,
        ov_config: Optional[dict] = None,
        low_memory: bool = False,
    ):
        """
        Args:
            model_dir: Directory with transformer.xml, text_encoder.xml, tokenizer.json and config.json.
            device: OpenVINO device for the transformer (CPU, GPU, ...).
            text_encoder_device: Device for the text encoder; defaults to ``device``.
            ov_config: Extra OpenVINO properties for ``compile_model``; they override the defaults.
            low_memory: Keep only one of the two models in memory at a time (about 7 GB peak
                instead of 14 GB for INT8). Both models are then recompiled every time the
                prompt changes. Off by default: each model is compiled once and reused.
        """
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.core = ov.Core()
        self.device = device
        self.text_encoder_device = text_encoder_device or device
        self.ov_config = dict(ov_config or {})
        self.low_memory = low_memory
        self.text_encoder = None
        self.transformer = None
        self._prompt_cache = None  # (prompts, encodings) of the last call
        # no scheduler object: the sigma schedule is fixed and stored in config.json
        self.scheduler = None
        self.safety_checker = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path] = DEFAULT_MODEL_ID,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        subfolder: str = "",
        force_download: bool = False,
        **kwargs,
    ) -> "OVIrisTurboPipeline":
        """Load from a local directory or a Hugging Face Hub repo id, like a diffusers pipeline.

        Args:
            model_id: A local folder (e.g. ``ov_model/iris-3b-turbo-int8``) or a Hub repo id
                (default ``rupeshs/iris-3b-int8-ov``). A path that is not an existing folder is
                treated as a repo id.
            local_files_only: Hub only: use the files already in the Hugging Face cache and do
                not go online. Fails if the model was never downloaded.
            cache_dir, revision, token, force_download: Passed to ``snapshot_download``.
                ``use_auth_token`` is accepted as an alias of ``token``.
            subfolder: Folder inside the local directory or repo that holds the model files.

        Only the model files are downloaded, not the README or anything else in the repo.
        Remaining keyword arguments (``device``, ``ov_config``, ...) go to the constructor.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)
        token = token if token is not None else use_auth_token
        subfolder = str(subfolder or "").strip("/\\")

        if Path(model_id).is_dir():
            model_dir = Path(model_id) / subfolder
        else:
            from huggingface_hub import snapshot_download

            prefix = f"{subfolder}/" if subfolder else ""
            snapshot_dir = snapshot_download(
                repo_id=str(model_id),
                allow_patterns=[prefix + name for name in MODEL_FILES],
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                force_download=force_download,
            )
            model_dir = Path(snapshot_dir) / subfolder

        missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{model_dir} is not an Iris Turbo OpenVINO model folder: missing {', '.join(missing)}. "
                "Expected the files written by convert_iris_turbo_ov.py."
            )
        return cls(model_dir, **kwargs)

    @property
    def height(self) -> int:
        return self.config["height"]

    @property
    def width(self) -> int:
        return self.config["width"]

    def to(self, device: str) -> "OVIrisTurboPipeline":
        if device != self.device:
            self.device = self.text_encoder_device = device
            self.text_encoder = None
            self.transformer = None
        return self

    def reshape(
        self,
        batch_size: int = -1,
        height: int = -1,
        width: int = -1,
        num_images_per_prompt: int = -1,
    ) -> "OVIrisTurboPipeline":
        """Check the requested size. The IRs are static (the position tables are
        baked in at export), so there is nothing to reshape."""
        self._check_size(
            height if height > 0 else self.height, width if width > 0 else self.width
        )
        return self

    def compile(self) -> None:
        """Load both models now. In low-memory mode they are loaded on demand instead."""
        if self.low_memory:
            return
        if self.text_encoder is None:
            self.text_encoder = self._compile("text_encoder", self.text_encoder_device)
        if self.transformer is None:
            self.transformer = self._compile("transformer", self.device)

    def _check_size(self, height: int, width: int) -> None:
        if (height, width) != (self.height, self.width):
            raise ValueError(
                f"This Iris Turbo model was exported for {self.width}x{self.height} (width x height) "
                f"and cannot generate {width}x{height}. Set the image size to {self.width}x{self.height}, "
                "or re-export the model with --height/--width."
            )

    def _compile(self, name: str, device: str) -> ov.CompiledModel:
        properties = {"PERFORMANCE_HINT": "LATENCY"}
        if device != "CPU":
            # RMSNorm squares activations; keep FP32 on GPU to avoid FP16 overflow
            properties["INFERENCE_PRECISION_HINT"] = "f32"
            properties["CACHE_DIR"] = str(self.model_dir / "cache")
        properties.update(self.ov_config)
        start = time.perf_counter()
        compiled = self.core.compile_model(
            self.model_dir / f"{name}.xml", device, properties
        )
        print(f"compiled {name} on {device} in {time.perf_counter() - start:.1f}s")
        return compiled

    def _get_sigmas(
        self, num_inference_steps: Optional[int], sigmas: Optional[List[float]]
    ) -> List[float]:
        schedule = self.config["sigmas"]
        if sigmas is not None:
            sigmas = [float(s) for s in sigmas]
            if (
                not sigmas
                or sigmas[0] != 1.0
                or sigmas[-1] <= 0
                or any(a <= b for a, b in zip(sigmas, sigmas[1:]))
            ):
                raise ValueError(
                    "sigmas must start at 1.0 (pure noise) and strictly decrease, staying above 0"
                )
            return sigmas
        if num_inference_steps is None or num_inference_steps == len(schedule):
            return schedule
        if not 1 <= num_inference_steps < len(schedule):
            raise ValueError(
                f"num_inference_steps must be between 1 and {len(schedule)}: "
                f"Iris Turbo is distilled to {len(schedule)} steps, got {num_inference_steps}."
            )
        # fewer steps: keep the first and last noise levels and spread the rest evenly
        keep = (
            np.linspace(0, len(schedule) - 1, num_inference_steps).round().astype(int)
        )
        return [schedule[i] for i in keep]

    def encode_prompt(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """Caption states [1, T, 12, dim] and their mask [1, T].

        The chat-template suffix goes after the (possibly truncated) caption,
        exactly as ``iris3b.text.qwen3_vl`` assembles it.
        """
        cfg = self.config
        prefix, suffix = cfg["prefix_ids"], cfg["suffix_ids"]
        budget = cfg["max_length"] - len(suffix)
        caption = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(caption) > budget:
            print(f"warning: prompt is {len(caption)} tokens, truncated to {budget}")
        used = prefix + caption[:budget] + suffix
        seq_len = len(prefix) + cfg["max_length"]
        input_ids = np.full((1, seq_len), cfg["pad_id"], dtype=np.int64)
        attention_mask = np.zeros((1, seq_len), dtype=np.int64)
        input_ids[0, : len(used)] = used
        attention_mask[0, : len(used)] = 1

        if self.text_encoder is None:
            self.text_encoder = self._compile("text_encoder", self.text_encoder_device)
        states = self.text_encoder(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )[0]
        return states, attention_mask[:, len(prefix) :]

    def _encode_prompts(
        self, prompts: List[str]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        # same prompts as the last call (e.g. a new seed): the text encoder is not needed
        if self._prompt_cache is not None and self._prompt_cache[0] == prompts:
            return self._prompt_cache[1]
        if self.low_memory:
            self.transformer = None  # make room for the text encoder
        encodings = [self.encode_prompt(prompt) for prompt in prompts]
        if self.low_memory:
            self.text_encoder = None
        self._prompt_cache = (prompts, encodings)
        return encodings

    def _sample(
        self,
        states: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        sigmas: List[float],
    ) -> np.ndarray:
        """Deterministic few-step flow sampler (``iris3b.sampling.generate_few_step``).

        Returns the image as [1, C, H, W] in about [-1, 1].
        """
        cfg = self.config
        shape = (1, cfg["in_channels"], cfg["height"], cfg["width"])
        x = rng.standard_normal(shape, dtype=np.float32)
        clean = x
        for i, sigma in enumerate(sigmas):
            start = time.perf_counter()
            t_freq = timestep_features(
                sigma * cfg["num_train_timesteps"],
                cfg["timestep_freq_dim"],
                cfg["timestep_max_period"],
            )
            out = self.transformer(
                {"x": x, "t_freq": t_freq, "y": states, "y_mask": mask}
            )[0]
            clean = out if cfg["prediction"] == "x" else x - sigma * out
            if i + 1 < len(sigmas):
                # move the clean prediction to the next noise level, reusing the implied noise
                implied_noise = (x - (1 - sigma) * clean) / sigma
                nxt = sigmas[i + 1]
                x = ((1 - nxt) * clean + nxt * implied_noise).astype(np.float32)
            print(
                f"step {i + 1}/{len(sigmas)} (sigma {sigma:.3f}) {time.perf_counter() - start:.1f}s"
            )
        return clean

    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[Seed, List[Seed]]] = None,
        output_type: str = "pil",
    ) -> OVIrisTurboPipelineOutput:
        """
        Args:
            prompt: One prompt or a list of prompts.
            height, width: Must match the exported size; default to it.
            num_inference_steps: Defaults to the distilled schedule (4 steps). Fewer steps use a
                subset of it and give a rougher image; more are not supported.
            sigmas: Explicit noise schedule, one model call per entry. Overrides
                ``num_inference_steps``. Anything but the distilled schedule is outside what
                Turbo was trained for.
            num_images_per_prompt: Images per prompt, generated one after another (batch is 1).
            generator: A seed or ``numpy.random.Generator``, or a list with one per image.
            output_type: "pil" for PIL images, "np" for a float array [N, H, W, 3] in [0, 1].
        """
        if output_type not in ("pil", "np"):
            raise ValueError(f"output_type must be 'pil' or 'np', got {output_type!r}")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        self._check_size(height or self.height, width or self.width)
        sigmas = self._get_sigmas(num_inference_steps, sigmas)

        total = len(prompts) * num_images_per_prompt
        if isinstance(generator, (list, tuple)):
            if len(generator) != total:
                raise ValueError(f"Got {len(generator)} generators for {total} images")
            generators = [np.random.default_rng(g) for g in generator]
        else:
            generators = [np.random.default_rng(generator)] * total

        encodings = self._encode_prompts(prompts)
        if self.transformer is None:
            self.transformer = self._compile("transformer", self.device)

        images = []
        for i, rng in enumerate(generators):
            states, mask = encodings[i // num_images_per_prompt]
            clean = self._sample(states, mask, rng, sigmas)
            images.append(
                (np.clip(clean[0], -1, 1) + 1).transpose(1, 2, 0)
            )  # HWC in [0, 2]

        if output_type == "np":
            return OVIrisTurboPipelineOutput(images=np.stack(images) / 2)
        return OVIrisTurboPipelineOutput(
            images=[
                Image.fromarray((image * 127.5).round().astype(np.uint8))
                for image in images
            ]
        )
