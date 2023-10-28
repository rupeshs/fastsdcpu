### FastSD CPU :sparkles:

Faster version of stable diffusion running on CPU.

Took 10 seconds to generate single 512x512 image on Core i7-12700(With OpenVINO).

Based on [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model) .

![Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/fastsdcpu-screenshot.png)

## Features
- Supports 256,512,768 image sizes
- Supports Windows and Linux
- Saves images
- Settings to control,steps,guidance and seed
- Added safety checker setting
- Maximum inference steps increased to 25
- OpenVINO support

## OpenVINO support

Thanks [deinferno](https://github.com/deinferno) for the OpenVINO model contribution.
Currently, it is tested on Windows only,by default it is disabled.
We found a 50% speed improvement using OpenVINO.It 
Took 10 seconds to generate single 512x512 image on Core i7-12700

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
- To start fastsdcpu double click `start.bat`

## FastSD CPU on Linux

Ensure that you have Python 3.8 or higher version installed.


- Clone/download this repo
- In the terminal enter into fastsdcpu directory
- Run the following command

  `chmod +x install.sh`

  `./install.sh`
- To start FastSD CPU run

  `chmod +x start.sh`

  `./start.sh`

## Raspberry PI 4 support

Thanks WGNW_MGM for Raspberry PI 4 testing.FastSD CPU worked without problems.
System configuration - Raspberry Pi 4 with 4GB RAM, 8GB of SWAP memory.

### License

The fastsdcpu project is available as open source under the terms of the [MIT license](https://github.com/rupeshs/fastsdcpu/blob/main/LICENSE)
