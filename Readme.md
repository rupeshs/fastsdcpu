# FastSD CPU :sparkles:

FastSD CPU is a faster version of Stable Diffusion on CPU. Based on [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model).
The application available as :
- Desktop GUI (Qt)
- WebUI 
- CLI (CommandLine Interface)

Took 10 seconds to generate single 512x512 image on Core i7-12700(With OpenVINO).

## Supported platforms
 - Windows
 - Linux
 - Raspberry PI 4
 - Mac (Need testers)

![FastSD CPU Desktop GUI Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastsdcpu-gui.jpg)


## Features
- Supports 256,512,768 image sizes
- Supports Windows and Linux
- Saves images and diffusion setting used to generate the image
- Settings to control,steps,guidance and seed
- Added safety checker setting
- Maximum inference steps increased to 25
- Added OpenVINO support
- Added web UI 
- Added CommandLine Interface
- Fixed OpenVINO image reproducibility issue
- Fixed OpenVINO high RAM usage,thanks [deinferno](https://github.com/deinferno) 
- Added more than one image support
- Stores and loads settings

## OpenVINO support

Thanks [deinferno](https://github.com/deinferno) for the OpenVINO model contribution.
We can get 2X speed improvement when using OpenVINO. 

## LCM Models 

Currently LCM model(Dreamshaper_v7) is supported (Diffuser format).

- https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
- https://huggingface.co/deinferno/LCM_Dreamshaper_v7-openvino

## FastSD CPU on Windows
:exclamation:**You must have a working Python installation.(Recommended : Python 3.10 or 3.11 )**

Clone/download this repo or download release.

### Installation

 - Double click `install.bat`  (It will take some time to install,depending on your internet speed.)

### Run
You can run in desktop GUI mode or web UI mode.
#### Desktop GUI
- To start fastsdcpu double click `start.bat`
#### Web UI 
- To start fastsdcpu double click `start-webui.bat`
## FastSD CPU on Linux

Ensure that you have Python 3.8 or higher version installed.

- Clone/download this repo
- In the terminal enter into fastsdcpu directory
- Run the following command

  `chmod +x install.sh`

  `./install.sh`
- To start FastSD CPU run

  `chmod +x start.sh`

#### To start Desktop GUI
  `./start.sh`
#### To start Desktop GUI
  `./start-webui.sh`

## Web UI screenshot

![FastSD CPU WebUI Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastcpu-webui.png)

## Google Colab
Due to the limitation of using CPU/OpenVINO inside colab currently using GPU.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SuAqskB-_gjWLYNRFENAkIXZ1aoyINqL?usp=sharing)

### Run
You can run in desktop GUI mode or web UI mode.
#### Desktop GUI
- To start fastsdcpu double click `start.bat`

## CLI mode (Advanced users)

![FastSD CPU CLI Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/docs/images/fastcpu-cli.png)

 Open the terminal and enter into fastsdcpu folder.
 Activate virtual environment using the command:
#### Windows users :
 (Suppose it is in the d drive "d:\fastsdcpu")
  `d:\fastsdcpu\env\Scripts\activate.bat`

#### Posix users:
  `source env/bin/activate`

Start CLI by `src/app.py -h`
## Raspberry PI 4 support

Thanks WGNW_MGM for Raspberry PI 4 testing.FastSD CPU worked without problems.
System configuration - Raspberry Pi 4 with 4GB RAM, 8GB of SWAP memory.

### License

The fastsdcpu project is available as open source under the terms of the [MIT license](https://github.com/rupeshs/fastsdcpu/blob/main/LICENSE)
