# FastSD CPU — Fork for Learning Image Generation Internals

> **This is a personal fork of [rupeshs/fastsdcpu](https://github.com/rupeshs/fastsdcpu).**
> The goal of this fork is to understand how AI image generation works internally and to provide a clean CLI tool for running image generation from the terminal.

FastSD CPU generates images on CPU (no GPU required) using distilled diffusion models such as [Latent Consistency Models (LCM)](https://github.com/luosiallen/latent-consistency-model) and SD Turbo. It is built on top of the 🤗 [Diffusers](https://github.com/huggingface/diffusers) library.

---

## Table of Contents

- [How Image Generation Works Internally](#how-image-generation-works-internally)
- [CLI Quick-Start](#cli-quick-start)
- [CLI Reference](#cli-reference)
- [Full App Installation](#full-app-installation)
- [Memory Requirements](#memory-requirements)
- [License](#license)

---

## How Image Generation Works Internally

Understanding this pipeline is the main purpose of this fork. Here is what happens step by step when you run `generate.py --prompt "a cat"`:

### 1. Text Encoding (CLIP)
Your text prompt is tokenised and passed through a **CLIP text encoder** (or T5 for Flux models). The encoder converts the words into a dense numerical vector — a *text embedding* — that captures the semantic meaning of the description.

### 2. Latent Noise Initialisation
Instead of working directly in pixel space (which is huge), diffusion models operate in a compressed **latent space** produced by a Variational Auto-Encoder (VAE). Generation starts by sampling a block of random Gaussian noise in that latent space.

### 3. Iterative Denoising (U-Net + LCM Scheduler)
The core of generation is a **U-Net** neural network that predicts the noise present in the current latent. Normally this requires 20–50 denoising steps. LCM (Latent Consistency Model) and Turbo models are *distilled* versions that can converge in **1–4 steps**, making CPU inference practical.

Each step:
1. The U-Net takes the noisy latent + text embedding + timestep as input.
2. It predicts the noise component.
3. The **LCM scheduler** removes (a portion of) that noise and moves the latent closer to a clean image.

### 4. Latent → Pixel Image (VAE Decode)
Once denoising is complete, the latent tensor is decoded back into pixel space by the **VAE decoder**. An optional **Tiny AutoEncoder (TAESD)** can replace the full VAE decoder for faster (but slightly lower quality) decoding.

### 5. Safety Check & Save
An optional NSFW safety checker inspects the final image. The result is saved as a PNG (or JPEG) file.

```
Prompt ──► CLIP Encoder ──► text_embeddings
                                    │
Random noise ──► Latent space       │
                     │              ▼
                     └──► U-Net denoising (×N steps, LCM scheduler)
                                    │
                                    ▼
                           Decoded latent ──► VAE Decoder ──► Image file
```

### Key source files

| File | Role |
|---|---|
| `src/backend/lcm_text_to_image.py` | Orchestrates the full pipeline (init + generate) |
| `src/backend/pipelines/lcm.py` | Loads the correct Diffusers pipeline |
| `src/context.py` | High-level `generate_text_to_image` entry point |
| `src/app.py` | CLI argument parsing; dispatches to GUI / WebUI / CLI |
| `generate.py` | Simplified standalone CLI script (this fork) |

---

## CLI Quick-Start

`generate.py` is a self-contained script that runs image generation straight from your terminal without needing to understand the rest of the app.

### 1. Prerequisites

- Python **3.10** or higher
- `pip` (comes with Python)

### 2. Create and activate a virtual environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install diffusers transformers accelerate Pillow torch
```

> These are the only packages needed to run `generate.py`. They are a subset of the full `requirements.txt`.

### 4. Generate your first image

```bash
python generate.py --prompt "a futuristic city at sunset"
```

The image is saved to the current directory as `output.png`.

### 5. More examples

```bash
# Custom size and number of steps
python generate.py --prompt "a dragon in a forest" --width 768 --height 768 --steps 4

# Use a different model (any HuggingFace SD Turbo-compatible model)
python generate.py --prompt "portrait of an astronaut" --model "stabilityai/sdxl-turbo" --steps 1

# Set a seed for reproducibility
python generate.py --prompt "a red sports car" --seed 42

# Save to a custom path
python generate.py --prompt "snowy mountains" --output my_image.png
```

---

## CLI Reference

```
usage: generate.py [-h] --prompt PROMPT [--model MODEL] [--steps STEPS]
                   [--width WIDTH] [--height HEIGHT] [--guidance GUIDANCE]
                   [--seed SEED] [--output OUTPUT]

Simple CLI image generator powered by FastSD CPU / Diffusers

options:
  -h, --help           show this help message and exit
  --prompt PROMPT      Text description of the image to generate
  --model MODEL        HuggingFace model ID (default: stabilityai/sd-turbo)
  --steps STEPS        Number of inference steps (default: 1)
  --width WIDTH        Image width in pixels (default: 512)
  --height HEIGHT      Image height in pixels (default: 512)
  --guidance GUIDANCE  Guidance scale; >1 increases prompt adherence (default: 0.0)
  --seed SEED          Random seed for reproducibility (default: random)
  --output OUTPUT      Output file path (default: output.png)
```

---

## Full App Installation

The original FastSD CPU application includes a desktop GUI, a web UI, LoRA support, ControlNet, OpenVINO acceleration, and more. To install the full application:

### Linux / macOS

```bash
git clone https://github.com/edujbarrios/fastsdcpu
cd fastsdcpu
chmod +x install.sh && ./install.sh
./start.sh          # Desktop GUI
./start-webui.sh    # Web UI
```

### Windows

```
install.bat   → then   start.bat  (GUI)  or  start-webui.bat  (Web UI)
```

### Full CLI (advanced)

After installation, activate the virtual environment and run:

```bash
# Linux/macOS
source env/bin/activate
python src/app.py --prompt "a cat sitting on a bench"

# Interactive mode
python src/app.py -i

# All options
python src/app.py --help
```

---

## Memory Requirements

Minimum system RAM for running in different modes:

| Mode       | Min RAM |
|------------|---------|
| LCM        | 2 GB    |
| LCM-LoRA   | 4 GB    |
| OpenVINO   | 11 GB   |


---

## License

This project is licensed under the [MIT License](LICENSE). See the original upstream repository [rupeshs/fastsdcpu](https://github.com/rupeshs/fastsdcpu) for full contributor history and credits.
