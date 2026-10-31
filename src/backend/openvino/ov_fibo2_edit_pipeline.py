"""OpenVINO pipeline for Fibo 2.0 Edit Turbo (Bria AI), instruction based image editing in 4 steps"""

import functools
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import openvino as ov
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm.auto import tqdm

CONFIG_NAME = "fibo2_ov.json"
VISION_ENCODER = "vision_encoder/openvino_model.xml"
TEXT_ENCODER = "text_encoder/openvino_model.xml"
TRANSFORMER = "transformer/openvino_model.xml"
VAE_ENCODER = "vae_encoder/openvino_model.xml"
VAE_DECODER = "vae_decoder/openvino_model.xml"
TOKENIZER = "tokenizer"
FONTS = ("fonts/Aileron-Regular.otf", "fonts/DejaVuSans.ttf")

DOWNSAMPLING_RATIO = 16  # 8x VAE + 2x2 patchify
LATENT_CHANNELS = 128
DEFAULT_AREA = 1024 * 1024  # the size Fibo 2.0 was trained at, about one megapixel
MAX_IMAGES = 5
TEXT_ENCODER_IMAGE_SIZE = 768  # Qwen3-VL sees each image to edit at most this many pixels a side
MAX_LENGTH = 4096

# Where an image stands in a Qwen3-VL prompt, and the chat turn each prompt is wrapped in
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def resolve_model_dir(
    model_id: str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    hf_cache_dir: str | Path | None = None,
) -> Path:
    """A local directory as is; otherwise a Hugging Face Hub repo id, downloaded once into the Hub cache."""
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


def seeds_for(count: int, seed: int | None, generator) -> list[int]:
    """One seed per image: ``seed + i``; else from ``generator`` (an int, a list of ints, a torch.Generator or a numpy
    Generator); else drawn from numpy's global state."""
    if seed is not None:
        return [(seed + i) % 2**32 for i in range(count)]
    if isinstance(generator, (int, np.integer)):
        return [(int(generator) + i) % 2**32 for i in range(count)]
    if isinstance(generator, (list, tuple)):
        if len(generator) != count:
            raise ValueError(f"{len(generator)} generators for {count} images")
        return [seeds_for(1, None, g)[0] for g in generator]
    if hasattr(generator, "initial_seed"):  # torch.Generator
        return [(generator.initial_seed() + i) % 2**32 for i in range(count)]
    if isinstance(generator, np.random.Generator):
        return [int(s) for s in generator.integers(2**32, size=count)]
    return [int(s) for s in np.random.randint(2**32, size=count, dtype=np.int64)]


# Prompts: Fibo 2.0 reads an edit as one JSON object, a vision marker per image, the caption of the result when the
# prompt has one, then the edit instruction


def parse_prompt(prompt: str) -> tuple[dict | str | None, str | None]:
    """A prompt's caption, cleaned as Fibo 2.0's training cleaned captions, and its edit instruction. A prompt that
    isn't a JSON object comes back as it is, with no instruction."""
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction

    def empty(value):
        if isinstance(value, float):
            return math.isnan(value)
        return value is None or (isinstance(value, (str, dict, list)) and not value)

    def clean(value):
        if isinstance(value, dict):
            value = {k: clean(v) for k, v in value.items()}
            return {k: v for k, v in value.items() if not empty(v)}
        if isinstance(value, list):
            return [v for v in map(clean, value) if not empty(v)]
        return value

    keys = set(caption)
    scores = ("aesthetic_score", "preference_score")
    caption = {
        key: {k: v for k, v in value.items() if k not in scores}
        if isinstance(value, dict)
        else value
        for key, value in clean(caption).items()
        if key not in scores
    }
    caption = {key: value for key, value in caption.items() if value != {}}
    if keys == {"short_description", "text_render"}:
        caption = {
            "short_description": caption.get("short_description") or "",
            "text_render": caption.get("text_render") or [],
        }
    return caption or None, instruction


def caption_texts(caption) -> list[str]:
    """The texts of a caption's text_render entries that have one, in order."""
    entries = (caption.get("text_render") or []) if isinstance(caption, dict) else []
    return [
        entry["text"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]
    ]


def get_edit_prompt(prompt: str, num_images: int, text_images: bool = False) -> str:
    caption, instruction = parse_prompt(prompt)
    if instruction is None:  # the prompt is the instruction alone
        caption, instruction = None, prompt
    if text_images and isinstance(caption, dict):
        for entry in caption.get("text_render") or []:
            if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]:
                entry["image_tokens"] = VISION_MARKER
    if num_images == 1:
        edit = {"image": VISION_MARKER}
    else:
        edit = {f"image_{k}": VISION_MARKER for k in range(1, num_images + 1)}
    if caption:
        edit["structured_caption"] = caption
    edit["edit_instruction"] = instruction
    return json.dumps(edit, separators=(",", ":"), ensure_ascii=False)


# Images of a caption's text, drawn the way Fibo 2.0's training drew them: each text in one line, black on white, in
# Pillow's built-in font at size 20, and the characters it lacks in the first fallback font that has them

FONT_SIZE = 20
MAX_LINE_CHARS = 256
TEXT_PADDING = 2
LINE_SPACING = 4


@functools.lru_cache(maxsize=None)
def load_font(path: str) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)


@functools.lru_cache(maxsize=None)
def default_font() -> ImageFont.FreeTypeFont:
    try:
        font = ImageFont.load_default(FONT_SIZE)
    except TypeError:  # Pillow before 10.1 can't size its built-in font
        font = None
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise ImportError("Drawing the text of a caption needs Pillow 10.1 or later, built with FreeType")
    return font


@functools.lru_cache(maxsize=4096)
def has_glyph(font: ImageFont.FreeTypeFont, character: str) -> bool:
    # A font without a glyph for the character draws its "notdef" box, as it does for U+FFFE
    try:
        notdef, mask = font.getmask("\ufffe"), font.getmask(character)
    except Exception:
        return True
    return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))


def draw_text_image(text: str, fallback_fonts: Sequence[ImageFont.FreeTypeFont] = ()) -> Image.Image:
    font = default_font()
    line = " ".join(text.split())
    lines = [line[i : i + MAX_LINE_CHARS] for i in range(0, len(line), MAX_LINE_CHARS)] or [""]
    if all(ord(c) < 128 or has_glyph(font, c) for c in line):
        return _draw_one_font(lines, font)
    return _draw_mixed(lines, [font, *fallback_fonts])


def _draw_one_font(lines: list[str], font: ImageFont.FreeTypeFont) -> Image.Image:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    min_box = draw.textbbox((0, 0), "Ag", font=font)
    min_line_height = max(1, min_box[3] - min_box[1])
    sizes, max_width, total_height = [], 1, 0
    for line in lines:
        box = draw.textbbox((0, 0), line or " ", font=font)
        line_height = max(min_line_height, box[3] - box[1])
        sizes.append((box, line_height))
        max_width = max(max_width, box[2] - box[0])
        total_height += line_height
    total_height += LINE_SPACING * (len(lines) - 1)
    image = Image.new(
        "RGB", (max_width + TEXT_PADDING * 2, total_height + TEXT_PADDING * 2), "white"
    )
    draw = ImageDraw.Draw(image)
    y = TEXT_PADDING
    for line, (box, line_height) in zip(lines, sizes):
        draw.text((TEXT_PADDING - box[0], y - box[1]), line, font=font, fill=(0, 0, 0))
        y += line_height + LINE_SPACING
    return image


def _draw_mixed(lines: list[str], fonts: Sequence[ImageFont.FreeTypeFont]) -> Image.Image:
    def runs_of(text):
        runs = []
        for c in text:
            font = fonts[0]
            if not (ord(c) < 128 or has_glyph(fonts[0], c)):
                font = next((f for f in fonts[1:] if has_glyph(f, c)), fonts[0])
            if runs and runs[-1][1] is font:
                runs[-1][0] += c
            else:
                runs.append([c, font])
        return runs

    ascent, descent = fonts[0].getmetrics()
    line_height = max(1, ascent + descent)
    runs_per_line = [runs_of(line or " ") for line in lines]
    width = max(
        max(1, round(sum(font.getlength(run) for run, font in runs)))
        for runs in runs_per_line
    )
    height = line_height * len(lines) + LINE_SPACING * (len(lines) - 1)
    image = Image.new("RGB", (width + TEXT_PADDING * 2, height + TEXT_PADDING * 2), "white")
    draw = ImageDraw.Draw(image)
    y = TEXT_PADDING
    for runs in runs_per_line:
        x = float(TEXT_PADDING)
        for run, font in runs:
            draw.text((x, y + ascent), run, font=font, fill=(0, 0, 0), anchor="ls")
            x += font.getlength(run)
        y += line_height + LINE_SPACING
    return image


# Images to edit, as Fibo 2.0 was trained on them


def get_image_size(aspect_ratio: float, area: int = DEFAULT_AREA) -> tuple[int, int]:
    """The (width, height) Fibo 2.0 was trained at for an aspect ratio: of the sizes on the 16-px grid, at least 64 a
    side and within 10% of ``area``, the one that crops the least, then the one closest to ``area``."""
    multiple = DOWNSAMPLING_RATIO
    min_side = 4 * multiple
    best, best_key = None, None
    for width in range(min_side, 2 * area // min_side, multiple):
        # Only the heights on the grid within 10% of the area
        first = max(min_side, math.ceil(0.9 * area / width / multiple) * multiple)
        for height in range(first, 2 * area // width, multiple):
            if width * height > 1.1 * area:
                break
            if width * height < 0.9 * area:
                continue
            ratio = width / height
            crop = 1 - min(ratio / aspect_ratio, aspect_ratio / ratio)
            key = (round(crop, 9), abs(width * height - area), -width)
            if best_key is None or key < best_key:
                best, best_key = (width, height), key
    return best


def resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scale to cover width x height, keeping the aspect ratio, then crop the center."""
    image = image.convert("RGB")
    scale = max(height / image.height, width / image.width)
    resized = math.ceil(image.width * scale), math.ceil(image.height * scale)
    image = image.resize(resized, resample=Image.LANCZOS)
    left, top = (resized[0] - width) // 2, (resized[1] - height) // 2
    return image.crop((left, top, left + width, top + height))


def grey_out(image: Image.Image, mask: Image.Image) -> Image.Image:
    """The region ``mask`` marks (white on black) greyed out, through the mask made binary at 128."""
    grey = Image.new("RGB", image.size, (128, 128, 128))
    binary = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
    return Image.composite(grey, image.convert("RGB"), binary)


# Qwen3-VL's inputs: image patches, the vision tower's position tables and the language model's mRoPE


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_patches(image: Image.Image, processor: dict, merge: int) -> tuple[np.ndarray, int, int]:
    """Qwen2-VL image processing: resized to the patch grid, normalized, flattened to (grid_h * grid_w, 1536) in
    merge-block order. Returns the patches and the grid size."""
    patch, temporal = processor["patch_size"], processor["temporal_patch_size"]
    height, width = smart_resize(
        image.height,
        image.width,
        patch * merge,
        processor["size"]["shortest_edge"],
        processor["size"]["longest_edge"],
    )
    image = image.convert("RGB")
    if (width, height) != image.size:
        image = image.resize((width, height), resample=Image.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) * processor["rescale_factor"]
    mean = np.asarray(processor["image_mean"], np.float32)[:, None, None]
    std = np.asarray(processor["image_std"], np.float32)[:, None, None]
    pixels = (pixels - mean) / std
    grid_h, grid_w = height // patch, width // patch
    channels = pixels.shape[0]
    patches = pixels.reshape(channels, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)[:, :, :, :, :, None]
    patches = np.broadcast_to(patches, (*patches.shape[:5], temporal, patch, patch))
    return np.ascontiguousarray(patches.reshape(grid_h * grid_w, -1)), grid_h, grid_w


def merge_block_positions(grid_h: int, grid_w: int, merge: int) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) of each patch, in the merge-block order of image_patches."""
    rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    shape = (grid_h // merge, merge, grid_w // merge, merge)
    return (
        rows.reshape(shape).transpose(0, 2, 1, 3).reshape(-1),
        cols.reshape(shape).transpose(0, 2, 1, 3).reshape(-1),
    )


def vision_inputs(grid_h: int, grid_w: int, merge: int, side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The vision tower's bilinear (align_corners) taps into its side x side position table, their weights, and the
    rotary (row, col) of each patch."""
    rows, cols = merge_block_positions(grid_h, grid_w, merge)

    def taps(index, size):
        src = index.astype(np.float32) * np.float32(side - 1) / np.float32(max(size - 1, 1))
        floor = np.floor(src)
        offsets = np.arange(2)
        raw = floor.astype(np.int64)[:, None] + offsets
        weights = np.clip(1 - np.abs(src[:, None] - floor[:, None] - offsets), 0, None)
        return np.clip(raw, 0, side - 1), weights.astype(np.float32)

    h_taps, h_weights = taps(rows, grid_h)
    w_taps, w_weights = taps(cols, grid_w)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices.astype(np.int64), weights, np.stack([rows, cols], axis=-1).astype(np.int64)


def mrope_position_ids(input_ids: np.ndarray, grids: Sequence[tuple[int, int]], image_token_id: int,
                       merge: int) -> np.ndarray:
    """Qwen3-VL's (3, L) temporal, row, col positions of one prompt: text tokens count up on all three, each image's
    tokens take its grid offset by the position it starts at."""
    is_image = input_ids == image_token_id
    positions, start, current, grid_index = [], 0, 0, 0
    while start < len(input_ids):
        end = start
        while end < len(input_ids) and is_image[end] == is_image[start]:
            end += 1
        if is_image[start]:
            grid_h, grid_w = grids[grid_index][0] // merge, grids[grid_index][1] // merge
            grid_index += 1
            rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
            positions.append(np.stack([np.zeros(rows.size, np.int64), rows.reshape(-1), cols.reshape(-1)]) + current)
            current += max(grid_h, grid_w)
        else:
            positions.append(np.broadcast_to(np.arange(end - start) + current, (3, end - start)))
            current += end - start
        start = end
    return np.concatenate(positions, axis=1)


def mrope_cos_sin(position_ids: np.ndarray, head_dim: int, theta: float,
                  section: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved mRoPE of (3, L) positions -> cos, sin (1, L, head_dim)."""
    inv_freq = (1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)).astype(np.float32)
    freqs = position_ids.astype(np.float32)[..., None] * inv_freq
    interleaved = freqs[0].copy()
    for axis, offset in ((1, 1), (2, 2)):
        index = slice(offset, section[axis] * 3, 3)
        interleaved[..., index] = freqs[axis][..., index]
    emb = np.concatenate([interleaved, interleaved], axis=-1)[None]
    return np.cos(emb), np.sin(emb)


# The transformer's inputs


def gist_count(text_length: int, config: dict) -> int:
    """The number of gist tokens for a prompt of ``text_length`` tokens, from 256 to 512 in steps of 64."""
    span = config["gist_max_text_len"] - config["gist_min_text_len"]
    ratio = min(1.0, max(0.0, (float(text_length) - config["gist_min_text_len"]) / span))
    raw = config["min_num_gist_tokens"] + ratio * (config["max_num_gist_tokens"] - config["min_num_gist_tokens"])
    quantized = int(math.ceil(raw / config["gist_step"])) * config["gist_step"]
    return max(config["min_num_gist_tokens"], min(config["max_num_gist_tokens"], quantized))


def coordinate_grid(size: tuple[int, int, int], start: tuple[int, int, int]) -> np.ndarray:
    axes = [np.arange(s0, s0 + span) for s0, span in zip(start, size)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def rope_cos_sin(ids: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fibo 2.0's axial RoPE of (S, 3) ids -> cos, sin (S, 64)."""
    parts = []
    for axis, dim in enumerate(config["axes_dims"]):
        freqs = 1.0 / config["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
        parts.append(np.outer(ids[:, axis].astype(np.float64), freqs).astype(np.float32))
    angles = np.concatenate(parts, axis=-1)
    return np.cos(angles), np.sin(angles)


def sigmas_for(steps: int, image_seq_len: int, config: dict) -> np.ndarray:
    """FlowMatchEulerDiscreteScheduler's sigmas, 1 down to 1 / steps, shifted for the image size, then 0."""
    slope = (config["max_shift"] - config["base_shift"]) / (config["max_image_seq_len"] - config["base_image_seq_len"])
    mu = image_seq_len * slope + config["base_shift"] - slope * config["base_image_seq_len"]
    sigmas = np.linspace(1.0, 1 / steps, steps).astype(np.float32)
    if config.get("time_shift_type", "exponential") == "exponential":
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
    else:
        sigmas = mu / (mu + (1 / sigmas - 1))
    return np.append(sigmas.astype(np.float32), np.float32(0.0))


@dataclass
class Fibo2EditPipelineOutput:
    """As diffusers' pipeline outputs: ``images`` is a list of PIL images, or an array for output_type "np"
    (N, H, W, 3) in [0, 1] or "latent" (N, 128, H/16, W/16)."""

    images: list[Image.Image] | np.ndarray


class OVFibo2EditPipeline:
    """Fibo 2.0 Edit Turbo image editing on OpenVINO, called like a diffusers / optimum-intel OVDiffusionPipeline.

    >>> pipe = OVFibo2EditPipeline.from_pretrained("rupeshs/fibo2-edit-turbo-int4-ov")  # or a model directory
    >>> edited = pipe("replace the jacket with a red leather one", image=Image.open("photo.jpg"), seed=0).images[0]
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
        self.vision_encoder = self.text_encoder = self.transformer = None
        self.vae_encoder = self.vae_decoder = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / TOKENIZER)
        self.fonts = [str(self.model_dir / f) for f in FONTS if (self.model_dir / f).exists()]
        self.sigmas = np.zeros(0, np.float32)
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
    ) -> "OVFibo2EditPipeline":
        """``model_id`` is a model directory or a Hub repo id. ``kwargs`` go to the constructor: ``device``,
        ``ov_config``, ``compile``, and ``cache_dir``, the OpenVINO compiled-model cache."""
        return cls(resolve_model_dir(model_id, revision, token, local_files_only, hf_cache_dir), **kwargs)

    @property
    def device(self) -> str:
        return self._device

    @property
    def num_timesteps(self) -> int:
        return max(0, len(self.sigmas) - 1)

    def compile(self) -> None:
        """Compile the models for ``device``; done on first use when loaded with ``compile=False`` or after ``to``."""
        if self.transformer is not None:
            return
        # GPU runs in FP16, where the transformer's activations can overflow; scaling them down keeps the FP16 speed.
        # The language model fails to compile with ACTIVATIONS_SCALE_FACTOR on GPU (an f16 scale lands on its f32
        # hidden state concat) and drifts in plain FP16, so it runs in FP32: it runs once per call, not per step
        gpu = self._device != "CPU"
        scaled = {"ACTIVATIONS_SCALE_FACTOR": 8.0, **self.ov_config} if gpu else self.ov_config
        full_precision = {"INFERENCE_PRECISION_HINT": "f32", **self.ov_config} if gpu else self.ov_config
        compile = lambda name, config=self.ov_config: self.core.compile_model(
            self.model_dir / name, self._device, config
        ).create_infer_request()
        self.vision_encoder = compile(VISION_ENCODER)
        self.text_encoder = compile(TEXT_ENCODER, full_precision)
        self.transformer = compile(TRANSFORMER, scaled)
        self.vae_encoder = compile(VAE_ENCODER)
        self.vae_decoder = compile(VAE_DECODER)

    def clear_requests(self) -> None:
        self.vision_encoder = self.text_encoder = self.transformer = None
        self.vae_encoder = self.vae_decoder = None

    def set_progress_bar_config(self, **kwargs) -> None:
        self._progress_bar_config = kwargs

    def progress_bar(self, iterable=None, total: int | None = None) -> tqdm:
        return tqdm(iterable, total=total, **self._progress_bar_config)

    def reshape(self, batch_size: int = -1, height: int = -1, width: int = -1, num_images_per_prompt: int = -1,
                **kwargs) -> "OVFibo2EditPipeline":
        """As optimum-intel's reshape, but the models stay dynamic: this only checks the size."""
        for name, value in (("height", height), ("width", width)):
            if value != -1 and value % DOWNSAMPLING_RATIO:
                raise ValueError(f"{name} must be a multiple of {DOWNSAMPLING_RATIO}, not {value}")
        return self

    def to(self, device: str) -> "OVFibo2EditPipeline":
        """Move to another OpenVINO device ("CPU", "GPU", ...); the models recompile on the next call."""
        self._device = device.upper()
        self.clear_requests()
        return self

    def encode_image(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """One image -> its Qwen3-VL tokens (M, 2560), DeepStack features (3, M, 2560) and patch grid."""
        config = self.config["text_encoder"]
        merge = config["spatial_merge_size"]
        patches, grid_h, grid_w = image_patches(image, self.config["image_processor"], merge)
        indices, weights, positions = vision_inputs(grid_h, grid_w, merge, config["num_grid_per_side"])
        result = self.vision_encoder.infer([patches, indices, weights, positions])
        return result[0].copy(), result[1].copy(), (grid_h, grid_w)

    def encode_prompt(self, prompt: str, images: Sequence[Image.Image],
                      text_images: Sequence[Image.Image] = ()) -> np.ndarray:
        """An edit prompt, with Qwen3-VL's copies of the images to edit and the images of its caption's text ->
        the transformer's text bundles (1, 6, L, 7680)."""
        config = self.config["text_encoder"]
        merge = config["spatial_merge_size"]
        encoded = [self.encode_image(image) for image in [*images, *text_images]]
        text = PROMPT_TEMPLATE.format(get_edit_prompt(prompt, len(images), text_images=bool(text_images)))
        # As Qwen3-VL's processor does: each image's pad token repeated once per merged token
        parts = text.split(IMAGE_PAD)
        if len(parts) - 1 != len(encoded):
            raise ValueError("the prompt holds a vision marker of its own")
        text = parts[0] + "".join(
            IMAGE_PAD * embeds.shape[0] + part for (embeds, _, _), part in zip(encoded, parts[1:])
        )
        input_ids = np.asarray(self.tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
        if len(input_ids) > MAX_LENGTH:
            raise ValueError(f"the prompt and its images take {len(input_ids)} tokens, more than {MAX_LENGTH}")
        image_positions = np.flatnonzero(input_ids == config["image_token_id"]).astype(np.int64)
        position_ids = mrope_position_ids(input_ids, [grid for _, _, grid in encoded], config["image_token_id"], merge)
        cos, sin = mrope_cos_sin(position_ids, config["text_head_dim"], config["text_rope_theta"],
                                 config["mrope_section"])
        result = self.text_encoder.infer(
            [
                input_ids[None],
                cos,
                sin,
                np.concatenate([embeds for embeds, _, _ in encoded]),
                np.concatenate([deepstack for _, deepstack, _ in encoded], axis=1),
                image_positions,
            ]
        )
        return result[0].copy()

    def encode_references(self, images: Sequence[Image.Image]) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """The crops to edit -> their normalized latent tokens (1, M, 128) and token grid sizes."""
        tokens, grids = [], []
        for image in images:
            pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0
            tokens.append(self.vae_encoder.infer([pixels])[0].copy())
            grids.append((image.height // DOWNSAMPLING_RATIO, image.width // DOWNSAMPLING_RATIO))
        return np.concatenate(tokens, axis=1), grids

    def text_images_for(self, prompt: str, text_images, render_text_images: bool) -> list[Image.Image]:
        if text_images is not None:
            return list(text_images)
        texts = caption_texts(parse_prompt(prompt)[0])
        if not (render_text_images and texts):
            return []
        try:
            fonts = [load_font(path) for path in self.fonts]
            return [draw_text_image(text, fonts) for text in texts]
        except ImportError as error:
            warnings.warn(f"The caption's text is not shown to the text encoder: {error}")
            return []

    def __call__(
        self,
        prompt: str,
        image: Image.Image | Sequence[Image.Image],
        mask: Image.Image | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
        num_images_per_prompt: int = 1,
        generator=None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[["OVFibo2EditPipeline", int, float, dict], dict] | None = None,
        callback_on_step_end_tensor_inputs: Sequence[str] = ("latents",),
        seed: int | None = None,
        area: int = DEFAULT_AREA,
        text_images: Sequence[Image.Image] | None = None,
        render_text_images: bool = True,
    ) -> Fibo2EditPipelineOutput | tuple:
        """Edit ``image`` (or up to 5 images, referred to as ``<image_1>``, ``<image_2>``, ...) as ``prompt`` says:
        an edit instruction, or a JSON caption of the result with an "edit_instruction" key, as bria-vgl writes them.
        With ``mask`` (white on black), only that region of the single image is regenerated.

        The size defaults to the size Fibo 2.0 was trained at for the first image's aspect ratio, about ``area``
        pixels. The images of the caption's ``text_render`` texts are drawn as in training unless ``text_images``
        gives them. Seeds: see ``seeds_for``; ``last_seed`` is the first image's.

        ``guidance_scale`` and ``negative_prompt`` are accepted for diffusers compatibility and ignored: the turbo
        model runs without CFG.
        """
        if guidance_scale is not None and guidance_scale > 1:
            warnings.warn(f"guidance_scale={guidance_scale} is ignored: the turbo model runs without CFG")
        if negative_prompt:
            warnings.warn("negative_prompt is ignored: it needs CFG, which the turbo model runs without")
        if output_type not in ("pil", "np", "latent"):
            raise ValueError(f"output_type must be 'pil', 'np' or 'latent', not {output_type!r}")
        if set(callback_on_step_end_tensor_inputs) - {"latents"}:
            raise ValueError("callback_on_step_end_tensor_inputs supports only 'latents'")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a single string")
        images = [image] if isinstance(image, Image.Image) else list(image or ())
        if not 1 <= len(images) <= MAX_IMAGES:
            raise ValueError(f"Fibo 2.0 Edit edits 1 to {MAX_IMAGES} images, not {len(images)}")
        images = [ImageOps.exif_transpose(img).convert("RGB") for img in images]
        if mask is not None:
            if len(images) != 1:
                raise ValueError("mask needs exactly one image")
            if mask.size != images[0].size:
                raise ValueError(f"mask has size {mask.size} but the image has size {images[0].size}")
            images = [grey_out(images[0], mask)]

        if height is None and width is None:
            width, height = get_image_size(images[0].width / images[0].height, area)
        height = height or width
        width = width or height
        if width % DOWNSAMPLING_RATIO or height % DOWNSAMPLING_RATIO:
            raise ValueError(f"width and height must be multiples of {DOWNSAMPLING_RATIO}")
        if len(images) == 1:
            crops = [resize_and_crop(images[0], width, height)]
        else:
            crops = [resize_and_crop(img, *get_image_size(img.width / img.height, area)) for img in images]
        text_encoder_images = []
        for crop in crops:
            copy = crop.copy()
            copy.thumbnail((TEXT_ENCODER_IMAGE_SIZE, TEXT_ENCODER_IMAGE_SIZE))
            text_encoder_images.append(copy)

        steps = num_inference_steps or self.config["steps"]
        seeds = seeds_for(num_images_per_prompt, seed, generator)
        self.last_seed = seeds[0]
        self.timings = {"steps": []}
        self.compile()

        config = self.config["transformer"]
        prompt_embeds = self.encode_prompt(
            prompt, text_encoder_images, self.text_images_for(prompt, text_images, render_text_images)
        )
        context, context_grids = self.encode_references(crops)
        h, w = height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO
        # RoPE planes: the gist tokens 1 to G, the image max_num_gist_tokens + 1, then each context image
        plane = config["max_num_gist_tokens"] + 1
        ids = [coordinate_grid((1, h, w), (plane, 0, 0))]
        ids += [coordinate_grid((1, *grid), (plane + k + 1, 0, 0)) for k, grid in enumerate(context_grids)]
        image_cos, image_sin = rope_cos_sin(np.concatenate(ids), config)
        gist = gist_count(prompt_embeds.shape[2], config)
        gist_cos, gist_sin = rope_cos_sin(coordinate_grid((gist, 1, 1), (1, 0, 0)), config)
        self.sigmas = sigmas_for(steps, h * w, self.config["scheduler"])

        outputs = []
        for seed in seeds:
            x = np.random.default_rng(seed).standard_normal((1, h * w, LATENT_CHANNELS), dtype=np.float32)
            for step in self.progress_bar(range(steps)):
                start = time.perf_counter()
                sigma, sigma_next = self.sigmas[step], self.sigmas[step + 1]
                velocity = self.transformer.infer(
                    [
                        x,
                        context,
                        np.array([sigma], dtype=np.float32),
                        prompt_embeds,
                        image_cos,
                        image_sin,
                        gist_cos,
                        gist_sin,
                    ]
                )[0]
                x = x + np.float32(sigma_next - sigma) * velocity
                self.timings["steps"].append(time.perf_counter() - start)
                if callback_on_step_end is not None:
                    x = (callback_on_step_end(self, step, float(sigma), {"latents": x}) or {}).get("latents", x)

            latent = np.ascontiguousarray(x[0].reshape(h, w, LATENT_CHANNELS).transpose(2, 0, 1)[None])
            if output_type == "latent":
                outputs.append(latent[0])
                continue
            pixels = self.vae_decoder.infer([latent])[0][0]
            array = (np.clip(pixels / 2 + 0.5, 0.0, 1.0) * 255).round().astype(np.uint8).transpose(1, 2, 0)
            outputs.append(Image.fromarray(array))

        if output_type == "np":
            outputs = np.stack([np.asarray(result, dtype=np.float32) / 255.0 for result in outputs])
        elif output_type == "latent":
            outputs = np.stack(outputs)
        return Fibo2EditPipelineOutput(images=outputs) if return_dict else (outputs,)
