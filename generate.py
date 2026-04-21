"""
generate.py — Standalone CLI image generator for FastSD CPU (fork)

Generates an image from a text prompt using a distilled Stable Diffusion model
(SD Turbo by default) via the Hugging Face Diffusers library. No GPU required.

Usage:
    python generate.py --prompt "a futuristic city at sunset"
    python generate.py --prompt "a dragon" --steps 4 --width 512 --height 512
    python generate.py --help

Setup (first time):
    python -m venv venv
    source venv/bin/activate   # Windows: venv\\Scripts\\activate
    pip install diffusers transformers accelerate Pillow torch
"""

import argparse
import random
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple CLI image generator powered by FastSD CPU / Diffusers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text description of the image to generate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/sd-turbo",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of inference steps (1 is enough for Turbo/LCM models)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height in pixels",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=0.0,
        help="Guidance scale; set >1.0 to make output follow the prompt more strictly",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (omit for a random result)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image file path (.png or .jpg)",
    )
    return parser.parse_args()


def load_pipeline(model_id: str):
    """
    Load the Diffusers AutoPipeline for text-to-image generation.

    How it works internally:
      1. Downloads (or reads from cache) the model weights from HuggingFace.
      2. Constructs a pipeline containing:
           - A CLIP text encoder — converts your prompt into embeddings.
           - A U-Net — the neural network that iteratively denoises a latent.
           - A VAE decoder — converts the denoised latent back to pixels.
           - A scheduler — controls the denoising schedule (e.g. LCM or DDIM).
    """
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        print(
            "ERROR: Required packages are not installed.\n"
            "Run: pip install diffusers transformers accelerate Pillow torch"
        )
        sys.exit(1)

    print(f"Loading model: {model_id}")
    print("(This may take a while the first time — weights are downloaded from HuggingFace)\n")

    dtype = torch.float32  # CPU-safe; use torch.float16 if you have a GPU
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipeline.to("cpu")
    return pipeline


def generate(pipeline, args):
    """
    Run the generation pipeline.

    Step-by-step what happens here:
      1. The prompt is encoded into a text embedding vector via CLIP.
      2. A random latent tensor (noise) is initialised.
         If --seed is provided the same noise is produced every run.
      3. The U-Net denoises the latent over `steps` iterations guided by
         the text embedding and the LCM / DDIM scheduler.
      4. The VAE decoder maps the final clean latent to a PIL image.
    """
    import torch

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"Prompt  : {args.prompt}")
    print(f"Model   : {args.model}")
    print(f"Size    : {args.width}x{args.height}")
    print(f"Steps   : {args.steps}")
    print(f"Guidance: {args.guidance}")
    print(f"Seed    : {seed}")
    print("\nGenerating image…")

    result = pipeline(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance,
        generator=generator,
    )

    return result.images[0]


def save_image(image, output_path: str):
    out = Path(output_path)
    image.save(out)
    print(f"\nImage saved to: {out.resolve()}")


def main():
    args = parse_args()
    pipeline = load_pipeline(args.model)
    image = generate(pipeline, args)
    save_image(image, args.output)


if __name__ == "__main__":
    main()
