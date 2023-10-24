### FastSD CPU :sparkles:

Faster version of stable diffusion running on CPU.

Took 10 seconds to generate single 512x512 image on Core i7-12700(With OpenVINO).

Based on [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model) .

![Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/fastsdcpu-screenshot.png)

## Features
- Supports 256,512,768 image sizes
- Supports Windows and Linux
- Saves images
- Settings to control,steps,guidance,seed
- Added safety checker setting
- Maximum inference steps increased to 25
- OpenVINO support

## OpenVINO support

Thanks [deinferno](https://github.com/deinferno) for the OpenVINO model contribution.
Currently it is tested on Windows only,by default it is disabled.
We found 50% speed improvement using OpenVINO.
Took 10 seconds to generate single 512x512 image on Core i7-12700

## LCM Models 
Currently LCM model(Dreamshaper_v7) is supported(Diffuser format).
- https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
- https://huggingface.co/deinferno/LCM_Dreamshaper_v7-openvino

## FastSD CPU on Windows
:exclamation:**You must have a working Python installation.(Recommended : Python 3.10 or 3.11 )**

Clone/download this repo or download release.

### Installation
 - Double click `install.bat`  (It will take some time to install,depends on your internet speed)

### Run
- To start fastsdcpu double click `start.bat`

## FastSD CPU on Linux
Ensure that you have Python 3.8 or higher version installed.

- Clone/download this repo
- In the terminal enter into fastsdcpu directory
- Run the following command

  `chmod +x install.sh`

  `./install.sh`
- To start Fast SD CPU run

  `chmod +x start.sh`

  `./start.sh`
