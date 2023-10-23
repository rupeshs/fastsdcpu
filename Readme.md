### FastSD CPU

Faster version of stable diffusion running on CPU.

Took 21 seconds to generate single 512x512 image on Core i7-12700

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

## FastSD CPU on Windows

- Clone/download this repo or download release.

### To install 
 - Double click `install.bat`  (It will take some time to install,depends on your internet speed)

### To run
- To start fastsdcpu double click `start.bat`

## FastSD CPU on Linux
Ensure that you have Python 3.8 higher installed.

- Clone/download this repo
- In the terminal enter into fastsdcpu directory
- Run the following command

  `chmod +x install.sh`

  `./install.sh`
- To start fastsdcpu run

  `chmod +x start.sh`

  `./start.sh`