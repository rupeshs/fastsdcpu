# FastSD CPU :sparkles:

FastSD CPU is a faster version of Stable Diffusion on CPU. Based on [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model).
The following interfaces are available :
- Desktop GUI (Qt)
- WebUI 
- CLI (CommandLine Interface)

Using OpenVINO, it took 10 seconds to create a single 512x512 image on a Core i7-12700.

## Supported platforms
 - Windows
 - Linux
 - Mac
 - Raspberry PI 4
 
 
![FastSD CPU Desktop GUI Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastsdcpu-gui.jpg)

## Features
- Supports 256,512,768 image sizes.
- Supports Windows and Linux.
- Saves images and diffusion setting used to generate the image.
- Settings to control,steps,guidance and seed.
- Added safety checker setting.
- Maximum inference steps increased to 25.
- Added [OpenVINO](https://github.com/openvinotoolkit/openvino) support
- Added web UI.
- Added CommandLine Interface.(CLI)
- Fixed OpenVINO image reproducibility issue.
- Fixed OpenVINO high RAM usage,thanks [deinferno](https://github.com/deinferno) .
- Added multiple image generation support.
- Application settings.

## OpenVINO support

Thanks [deinferno](https://github.com/deinferno) for the OpenVINO model contribution.
We can get 2x speed improvement when using OpenVINO. 

## LCM Models 

Following LCM models are supported:

- LCM_Dreamshaper_v7 -https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7 by [Simian Luo](https://github.com/luosiallen)
- LCM_Dreamshaper_v7-openvino - https://huggingface.co/deinferno/LCM_Dreamshaper_v7-openvino by [deinferno](https://github.com/deinferno) 

## FastSD CPU on Windows
:exclamation:**You must have a working Python installation.(Recommended : Python 3.10 or 3.11 )**

Clone/download this repo or download release.

### Installation

 - Double click `install.bat`  (It will take some time to install,depending on your internet speed.)

### Run
You can run in desktop GUI mode or web UI mode.
#### Desktop GUI
- To start desktop GUI double click `start.bat`
#### Web UI 
- To start web UI, double click `start-webui.bat`
## FastSD CPU on Linux

Ensure that you have Python 3.8 or higher version installed.

- Clone/download this repo
- In the terminal, enter into fastsdcpu directory
- Run the following command

  `chmod +x install.sh`

  `./install.sh`

#### To start Desktop GUI

  `./start.sh`
#### To start Web UI

  `./start-webui.sh`


## FastSD CPU on Mac

![FastSD CPU running on Mac](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastsdcpu-mac-gui.jpg)

### Installation
Ensure that you have Python 3.8 or higher version installed.

- Clone/download this repo
- In the terminal, enter into fastsdcpu directory
- Run the following command

  `chmod +x install-mac.sh`

  `./install-mac.sh`

#### To start Desktop GUI

  `./start.sh`
#### To start Web UI

  `./start-webui.sh`

Thanks [Autantpourmoi ](https://github.com/Autantpourmoi) for Mac testing.

:exclamation:We don't support OpenVINO on Mac. 

If you want to increase image generation speed on Mac(M1/M2 chip) try this:

`export DEVICE=mps` and start app `start.sh`


## Web UI screenshot

![FastSD CPU WebUI Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastcpu-webui.png)

## Google Colab
Due to the limitation of using CPU/OpenVINO inside colab, we are using GPU with colab.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SuAqskB-_gjWLYNRFENAkIXZ1aoyINqL?usp=sharing)


## CLI mode (Advanced users)

![FastSD CPU CLI Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastcpu-cli.png)

 Open the terminal and enter into fastsdcpu folder.
 Activate virtual environment using the command:
#### Windows users :
 (Suppose FastSD CPU available in the directory "D:\fastsdcpu")
  `D:\fastsdcpu\env\Scripts\activate.bat`

#### Linux users:
  `source env/bin/activate`

Start CLI  `src/app.py -h`
## Raspberry PI 4 support

Thanks WGNW_MGM for Raspberry PI 4 testing.FastSD CPU worked without problems.
System configuration - Raspberry Pi 4 with 4GB RAM, 8GB of SWAP memory.

### License

The fastsdcpu project is available as open source under the terms of the [MIT license](https://github.com/rupeshs/fastsdcpu/blob/main/LICENSE)
