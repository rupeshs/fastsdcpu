# FastSD CPU :sparkles:[![Mentioned in Awesome OpenVINO](https://awesome.re/mentioned-badge-flat.svg)](https://github.com/openvinotoolkit/awesome-openvino)

<div align="center">
  <a href="https://trendshift.io/repositories/3957" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3957" alt="rupeshs%2Ffastsdcpu | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

FastSD CPU is a faster version of Stable Diffusion on CPU. Based on [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model) and
[Adversarial Diffusion Distillation](https://nolowiz.com/fast-stable-diffusion-on-cpu-using-fastsd-cpu-and-openvino/).

![FastSD CPU screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastsdcpu-webui.png)
The following interfaces are available :

- Desktop GUI, basic text to image generation (Qt,faster)
- WebUI (Advanced features,Lora,controlnet etc)
- CLI (CommandLine Interface)

🚀 Using __OpenVINO(SDXS-512-0.9)__, it took __0.82 seconds__ (__820 milliseconds__) to create a single 512x512 image on a __Core i7-12700__.

## Table of Contents

- [Supported&nbsp;Platforms](#Supported&nbsp;platforms)
- [Memory requirements](#memory-requirements)
- [Features](#features)
- [Benchmarks](#fast-inference-benchmarks)
- [OpenVINO Support](#openvino)
- [Installation](#installation)
- [Real-time text to image (EXPERIMENTAL)](#real-time-text-to-image)
- [Models](#models)
- [How to use Lora models](#useloramodels)
- [How to use controlnet](#usecontrolnet)
- [Android](#android)
- [Raspberry Pi 4](#raspberry)
- [Orange Pi 5](#orangepi)
- [API&nbsp;Support](#apisupport)
- [License](#license)
- [Contributors](#contributors)

## Supported platforms⚡️

FastSD CPU works on the following platforms:

- Windows
- Linux
- Mac
- Android + Termux
- Raspberry PI 4

## Memory requirements

Minimum system RAM requirement for FastSD CPU.

Model (LCM,OpenVINO): SD Turbo, 1 step, 512 x 512

Model (LCM-LoRA): Dreamshaper v8, 3 step, 512 x 512

| Mode                  | Min RAM       |
| --------------------- | ------------- |
| LCM                   | 2 GB          |
| LCM-LoRA              | 4 GB          |
| OpenVINO              | 11 GB         |

If we enable Tiny decoder(TAESD) we can save some memory(2GB approx) for example in OpenVINO mode memory usage will become 9GB.

:exclamation: Please note that guidance scale >1 increases RAM usage and slow inference speed.

## Features

- Desktop GUI, web UI and CLI
- Supports 256,512,768,1024 image sizes
- Supports Windows,Linux,Mac
- Saves images and diffusion setting used to generate the image
- Settings to control,steps,guidance and seed
- Added safety checker setting
- Maximum inference steps increased to 25
- Added [OpenVINO](https://github.com/openvinotoolkit/openvino) support
- Fixed OpenVINO image reproducibility issue
- Fixed OpenVINO high RAM usage,thanks [deinferno](https://github.com/deinferno)
- Added multiple image generation support
- Application settings
- Added Tiny Auto Encoder for SD (TAESD) support, 1.4x speed boost (Fast,moderate quality)
- Safety checker disabled by default
- Added SDXL,SSD1B - 1B LCM models
- Added LCM-LoRA support, works well for fine-tuned Stable Diffusion model 1.5  or SDXL models
- Added negative prompt support in LCM-LoRA mode
- LCM-LoRA models can be configured using text configuration file
- Added support for custom models for OpenVINO (LCM-LoRA baked)
- OpenVINO models now supports negative prompt (Set guidance >1.0)
- Real-time inference support,generates images while you type (experimental)
- Fast 2,3 steps inference
- Lcm-Lora fused models for faster inference
- Supports integrated GPU(iGPU) using OpenVINO (export DEVICE=GPU)
- 5.7x speed using OpenVINO(steps: 2,tiny autoencoder)
- Image to Image support (Use Web UI)
- OpenVINO image to image support
- Fast 1 step inference (SDXL Turbo)
- Added SD Turbo support
- Added image to image support for Turbo models (Pytorch and OpenVINO)
- Added image variations support
- Added 2x upscaler (EDSR and Tiled SD upscale (experimental)),thanks [monstruosoft](https://github.com/monstruosoft) for SD upscale
- Works on Android + Termux + PRoot
- Added interactive CLI,thanks [monstruosoft](https://github.com/monstruosoft)
- Added basic lora support to CLI and WebUI
- ONNX EDSR 2x upscale
- Add SDXL-Lightning support
- Add SDXL-Lightning OpenVINO support (int8)
- Add multilora support,thanks [monstruosoft](https://github.com/monstruosoft)
- Add basic ControlNet v1.1 support(LCM-LoRA mode),thanks [monstruosoft](https://github.com/monstruosoft)
- Add ControlNet annotators(Canny,Depth,LineArt,MLSD,NormalBAE,Pose,SoftEdge,Shuffle)
- Add SDXS-512 0.9 support
- Add SDXS-512 0.9 OpenVINO,fast 1 step inference (0.8 seconds to generate 512x512 image)
- Default model changed to SDXS-512-0.9
- Faster realtime image generation
- Add NPU device check
- Revert default model to SDTurbo
- Update realtime UI
- Add hypersd support
- 1 step fast inference support for SDXL and SD1.5
- Experimental support for single file Safetensors SD 1.5 models(Civitai models), simply add local model path to configs/stable-diffusion-models.txt file.
- Add REST API support

<a id="fast-inference-benchmarks"></a>

## Fast Inference Benchmarks

### 🚀 Fast 1 step inference with Hyper-SD

#### Stable diffuion 1.5

Works with LCM-LoRA mode.
Fast 1 step inference supported on `runwayml/stable-diffusion-v1-5` model,select `rupeshs/hypersd-sd1-5-1-step-lora` lcm_lora model from the settings.

#### Stable diffuion XL

Works with LCM and LCM-OpenVINO mode.

- *Hyper-SD SDXL 1 step* - [rupeshs/hyper-sd-sdxl-1-step](https://huggingface.co/rupeshs/hyper-sd-sdxl-1-step)

- *Hyper-SD SDXL 1 step OpenVINO* - [rupeshs/hyper-sd-sdxl-1-step-openvino-int8](https://huggingface.co/rupeshs/hyper-sd-sdxl-1-step-openvino-int8)

#### Inference Speed

Tested on Core i7-12700 to generate __768x768__ image(1 step).

| Diffusion Pipeline    | Latency       |
| --------------------- | ------------- |
| Pytorch               | 19s           |
| OpenVINO              | 13s           |
| OpenVINO + TAESDXL    | 6.3s          |

### Fastest 1 step inference (SDXS-512-0.9)

:exclamation:This is an experimental model, only text to image workflow is supported.

#### Inference Speed

Tested on Core i7-12700 to generate __512x512__ image(1 step).

__SDXS-512-0.9__

| Diffusion Pipeline    | Latency       |
| --------------------- | ------------- |
| Pytorch               | 4.8s          |
| OpenVINO              | 3.8s          |
| OpenVINO + TAESD      | __0.82s__     |

### 🚀 Fast 1 step inference (SD/SDXL Turbo - Adversarial Diffusion Distillation,ADD)

Added support for ultra fast 1 step inference using [sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) model

:exclamation: These SD turbo models are intended for research purpose only.

#### Inference Speed

Tested on Core i7-12700 to generate __512x512__ image(1 step).

__SD Turbo__

| Diffusion Pipeline    | Latency       |
| --------------------- | ------------- |
| Pytorch               | 7.8s          |
| OpenVINO              | 5s            |
| OpenVINO + TAESD      | 1.7s          |

__SDXL Turbo__

| Diffusion Pipeline    | Latency       |
| --------------------- | ------------- |
| Pytorch               | 10s           |
| OpenVINO              | 5.6s          |
| OpenVINO + TAESDXL    | 2.5s          |

### 🚀 Fast 2 step inference (SDXL-Lightning - Adversarial Diffusion Distillation)

SDXL-Lightning works with LCM and LCM-OpenVINO mode.You can select these models from app settings.

Tested on Core i7-12700 to generate __768x768__ image(2 steps).

| Diffusion Pipeline    | Latency       |
| --------------------- | ------------- |
| Pytorch               | 18s           |
| OpenVINO              | 12s           |
| OpenVINO + TAESDXL    | 10s           |

- *SDXL-Lightning* - [rupeshs/SDXL-Lightning-2steps](https://huggingface.co/rupeshs/SDXL-Lightning-2steps)

- *SDXL-Lightning OpenVINO* - [rupeshs/SDXL-Lightning-2steps-openvino-int8](https://huggingface.co/rupeshs/SDXL-Lightning-2steps-openvino-int8)

### 2 Steps fast inference (LCM)

FastSD CPU supports 2 to 3 steps fast inference using LCM-LoRA workflow. It works well with SD 1.5 models.

![2 Steps inference](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/2steps-inference.jpg)

### Benchmark scripts

To benchmark run the following batch file on Windows:

- `benchmark.bat` - To benchmark Pytorch
- `benchmark-openvino.bat` - To benchmark OpenVINO

Alternatively you can run benchmarks by passing `-b` command line argument in CLI mode.
<a id="openvino"></a>

## OpenVINO support

Fast SD CPU utilizes [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) to speed up the inference speed.
Thanks [deinferno](https://github.com/deinferno) for the OpenVINO model contribution.
We can get 2x speed improvement when using OpenVINO.
Thanks [Disty0](https://github.com/Disty0) for the conversion script.

### OpenVINO SDXL models

These are models converted to use directly use it with FastSD CPU. These models are compressed to int8 to reduce the file size (10GB to 4.4 GB) using [NNCF](https://github.com/openvinotoolkit/nncf)

- Hyper-SD SDXL 1 step - [rupeshs/hyper-sd-sdxl-1-step-openvino-int8](https://huggingface.co/rupeshs/hyper-sd-sdxl-1-step-openvino-int8)
- SDXL Lightning 2 steps - [rupeshs/SDXL-Lightning-2steps-openvino-int8](https://huggingface.co/rupeshs/SDXL-Lightning-2steps-openvino-int8)

### OpenVINO SD Turbo models

We have converted SD/SDXL Turbo models to OpenVINO for fast inference on CPU. These models are intended for research purpose only. Also we converted TAESDXL MODEL to OpenVINO and

- *SD Turbo OpenVINO* - [rupeshs/sd-turbo-openvino](https://huggingface.co/rupeshs/sd-turbo-openvino)
- *SDXL Turbo OpenVINO int8* - [rupeshs/sdxl-turbo-openvino-int8](https://huggingface.co/rupeshs/sdxl-turbo-openvino-int8)
- *TAESDXL OpenVINO* - [rupeshs/taesdxl-openvino](https://huggingface.co/rupeshs/taesdxl-openvino)

You can directly use these models in FastSD CPU.

### Convert SD 1.5 models to OpenVINO LCM-LoRA fused models

We first creates LCM-LoRA baked in model,replaces the scheduler with LCM and then converts it into OpenVINO model. For more details check [LCM OpenVINO Converter](https://github.com/rupeshs/lcm-openvino-converter), you can use this tools to convert any StableDiffusion 1.5 fine tuned models to OpenVINO.
<a id="real-time-text-to-image"></a>

## Real-time text to image (EXPERIMENTAL)

We can generate real-time text to images using FastSD CPU.

__CPU (OpenVINO)__

Near real-time inference on CPU using OpenVINO, run the `start-realtime.bat` batch file and open the link in browser (Resolution : 512x512,Latency : 0.82s on Intel Core i7)

Watch YouTube video :

[![IMAGE_ALT](https://img.youtube.com/vi/0XMiLc_vsyI/0.jpg)](https://www.youtube.com/watch?v=0XMiLc_vsyI)

## Models

To use single file [Safetensors](https://huggingface.co/docs/safetensors/en/index) SD 1.5 models(Civit AI) follow this [YouTube tutorial](https://www.youtube.com/watch?v=zZTfUZnXJVk). Use LCM-LoRA Mode for single file safetensors.

Fast SD supports LCM models and LCM-LoRA models.

### LCM Models

These models can be configured in `configs/lcm-models.txt` file.

### OpenVINO models

These are LCM-LoRA baked in models. These models can be configured in `configs/openvino-lcm-models.txt` file

### LCM-LoRA models

These models can be configured in `configs/lcm-lora-models.txt` file.

- *lcm-lora-sdv1-5* - distilled consistency adapter for [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- *lcm-lora-sdxl* - Distilled consistency adapter for [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- *lcm-lora-ssd-1b* - Distilled consistency adapter for [segmind/SSD-1B](https://huggingface.co/segmind/SSD-1B)

These models are used with Stablediffusion base models `configs/stable-diffusion-models.txt`.

:exclamation: Currently no support for OpenVINO LCM-LoRA models.

### How to add new LCM-LoRA models

To add new model follow the steps:
For example we will add `wavymulder/collage-diffusion`, you can give Stable diffusion 1.5 Or SDXL,SSD-1B fine tuned models.

1. Open `configs/stable-diffusion-models.txt` file in text editor.
2. Add the model ID `wavymulder/collage-diffusion`  or locally cloned path.

Updated file as shown below :

```Lykon/dreamshaper-8
Fictiverse/Stable_Diffusion_PaperCut_Model
stabilityai/stable-diffusion-xl-base-1.0
runwayml/stable-diffusion-v1-5
segmind/SSD-1B
stablediffusionapi/anything-v5
wavymulder/collage-diffusion
```

Similarly we can update `configs/lcm-lora-models.txt` file with lcm-lora ID.

### How to use LCM-LoRA models offline

Please follow the steps to run LCM-LoRA models offline :

- In the settings ensure that  "Use locally cached model" setting is ticked.
- Download the model for example `latent-consistency/lcm-lora-sdv1-5`
Run the following commands:

```
git lfs install
git clone https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
```

Copy the cloned model folder path for example "D:\demo\lcm-lora-sdv1-5" and update the `configs/lcm-lora-models.txt` file as shown below :

```
D:\demo\lcm-lora-sdv1-5
latent-consistency/lcm-lora-sdxl
latent-consistency/lcm-lora-ssd-1b
```

- Open the app and select the newly added local folder in the combo box menu.
- That's all!
<a id="useloramodels"></a>

## How to use Lora models

Place your lora models in "lora_models" folder. Use LCM or LCM-Lora mode.
You can download lora model (.safetensors/Safetensor) from [Civitai](https://civitai.com/) or [Hugging Face](https://huggingface.co/)
E.g: [cutecartoonredmond](https://civitai.com/models/207984/cutecartoonredmond-15v-cute-cartoon-lora-for-liberteredmond-sd-15?modelVersionId=234192)
<a id="usecontrolnet"></a>

## ControlNet support

We can use ControlNet in LCM-LoRA mode.

Download ControlNet models from [ControlNet-v1-1](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/tree/main).Download and place controlnet models in "controlnet_models" folder.

Use the medium size models (723 MB)(For example : <https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11p_sd15_canny_fp16.safetensors>)

## Installation

### FastSD CPU on Windows

![FastSD CPU Desktop GUI Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastsdcpu-gui.jpg)

:exclamation:__You must have a working Python installation.(Recommended : Python 3.10 or 3.11 )__

To install FastSD CPU on Windows run the following steps :

- Clone/download this repo or download [release](https://github.com/rupeshs/fastsdcpu/releases).
- Double click `install.bat`  (It will take some time to install,depending on your internet speed.)
- You can run in desktop GUI mode or web UI mode.

#### Desktop GUI

- To start desktop GUI double click `start.bat`

#### Web UI

- To start web UI double click `start-webui.bat`

### FastSD CPU on Linux

:exclamation:__Ensure that you have Python 3.9 or 3.10 or 3.11 version installed.__

- Clone/download this repo or download [release](https://github.com/rupeshs/fastsdcpu/releases).
- In the terminal, enter into fastsdcpu directory
- Run the following command

  `chmod +x install.sh`

  `./install.sh`

#### To start Desktop GUI

  `./start.sh`

#### To start Web UI

  `./start-webui.sh`

### FastSD CPU on Mac

![FastSD CPU running on Mac](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastsdcpu-mac-gui.jpg)

:exclamation:__Ensure that you have Python 3.9 or 3.10 or 3.11 version installed.__

Run the following commands to install FastSD CPU on Mac :

- Clone/download this repo or download [release](https://github.com/rupeshs/fastsdcpu/releases).
- In the terminal, enter into fastsdcpu directory
- Run the following command

  `chmod +x install-mac.sh`

  `./install-mac.sh`

#### To start Desktop GUI

  `./start.sh`

#### To start Web UI

  `./start-webui.sh`

Thanks [Autantpourmoi](https://github.com/Autantpourmoi) for Mac testing.

:exclamation:We don't support OpenVINO on Mac (M1/M2/M3 chips, but *does* work on Intel chips).

If you want to increase image generation speed on Mac(M1/M2 chip) try this:

`export DEVICE=mps` and start app `start.sh`

#### Web UI screenshot

![FastSD CPU WebUI Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastcpu-webui.png)

### Google Colab

Due to the limitation of using CPU/OpenVINO inside colab, we are using GPU with colab.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SuAqskB-_gjWLYNRFENAkIXZ1aoyINqL?usp=sharing)

### CLI mode (Advanced users)

![FastSD CPU CLI Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastcpu-cli.png)

 Open the terminal and enter into fastsdcpu folder.
 Activate virtual environment using the command:

##### Windows users

 (Suppose FastSD CPU available in the directory "D:\fastsdcpu")
  `D:\fastsdcpu\env\Scripts\activate.bat`

##### Linux users

  `source env/bin/activate`

Start CLI  `src/app.py -h`

<a id="android"></a>

## Android (Termux + PRoot)

FastSD CPU running on Google Pixel 7 Pro.

![FastSD CPU Android Termux Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastsdcpu-android-termux-pixel7.png)

### 1. Prerequisites

First you have to [install Termux](https://wiki.termux.com/wiki/Installing_from_F-Droid) and [install PRoot](https://wiki.termux.com/wiki/PRoot). Then install and login to Ubuntu in PRoot.

### 2. Install FastSD CPU

Run the following command to install without Qt GUI.

 `proot-distro login ubuntu`

 `./install.sh --disable-gui`

 After the installation you can use WebUi.

  `./start-webui.sh`

  Note : If you get `libgl.so.1` import error run `apt-get install ffmpeg`.

  Thanks [patienx](https://github.com/patientx) for this guide  [Step by step guide to installing FASTSDCPU on ANDROID](https://github.com/rupeshs/fastsdcpu/discussions/123)

Another step by step guide to run FastSD on Android is [here](https://nolowiz.com/how-to-install-and-run-fastsd-cpu-on-android-temux-step-by-step-guide/)

<a id="raspberry"></a>

## Raspberry PI 4 support

Thanks [WGNW_MGM] for Raspberry PI 4 testing.FastSD CPU worked without problems.
System configuration - Raspberry Pi 4 with 4GB RAM, 8GB of SWAP memory.

<a id="orangepi"></a>

## Orange Pi 5 support

Thanks [khanumballz](https://github.com/khanumballz) for testing FastSD CPU with Orange PI 5.
[Here is a video of FastSD CPU running on Orange Pi 5](https://www.youtube.com/watch?v=KEJiCU0aK8o).

<a id="apisupport"></a>

## API support

![FastSD CPU API documentation](https://raw.githubusercontent.com/rupeshs/fastsdcpu/add-basic-api-support/docs/images/fastsdcpu-api.png)

FastSD CPU supports basic API endpoints. Following API endpoints are available :

- /api/info - To get system information
- /api/config - Get configuration
- /api/models - List all available models
- /api/generate - Generate images (Text to image,image to image)

To start FastAPI in webserver mode run:
``python src/app.py --api``

or use  `start-webserver.sh` for Linux and  `start-webserver.bat` for Windows.

Access API documentation locally at <http://localhost:8000/api/docs> .

Generated image is JPEG image encoded as base64 string.
In the image-to-image mode input image should be encoded as base64 string.

To generate an image a minimal request `POST /api/generate` with body :

```
{
    "prompt": "a cute cat",
    "use_openvino": true
}
```

## Known issues

- TAESD will not work with OpenVINO image to image workflow

## License

The fastsdcpu project is available as open source under the terms of the [MIT license](https://github.com/rupeshs/fastsdcpu/blob/main/LICENSE)

## Disclaimer

Users are granted the freedom to create images using this tool, but they are obligated to comply with local laws and utilize it responsibly. The developers will not assume any responsibility for potential misuse by users.

## Contributors

<a href="https://github.com/rupeshs/fastsdcpu/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=rupeshs/fastsdcpu" />
</a>
