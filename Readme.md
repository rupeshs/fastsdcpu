### FastSD CPU :sparkles:

FastSD CPU is a faster version of stable diffusion running on CPU, capable of generating a single 512x512 image in just 10 seconds on a Core i7-12700 (with OpenVINO acceleration).

It is based on [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model).

![Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/fastsdcpu-screenshot.png)

## Features
- Supports 256, 512, and 768 image sizes.
- Compatible with both Windows and Linux.
- Ability to save generated images.
- Adjustable settings for controlling steps, guidance, and seed.
- Includes a safety checker setting.
- Maximum inference steps have been increased to 25.
- OpenVINO support for enhanced performance.

## OpenVINO Support

Special thanks to [deinferno](https://github.com/deinferno) for contributing the OpenVINO model. OpenVINO support is currently tested on Windows only and is disabled by default, offering a significant 50% speed improvement. It took only 10 seconds to generate a single 512x512 image on a Core i7-12700.

## LCM Models 

FastSD CPU supports the LCM model (Dreamshaper_v7) in Diffuser format:

- [SimianLuo/LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)
- [deinferno/LCM_Dreamshaper_v7-openvino](https://huggingface.co/deinferno/LCM_Dreamshaper_v7-openvino)

## FastSD CPU on Windows
:exclamation: **You must have a working Python installation. (Recommended versions: Python 3.10 or 3.11)**

To get started on Windows:
1. Clone/download this repository or download the release.
2. Installation
    - Double-click `install.bat` (Note: Installation time may vary based on your internet speed).

3. Running
    - To start FastSD CPU, simply double-click `start.bat`.

## FastSD CPU on Linux

To run FastSD CPU on Linux:
1. Ensure that you have Python 3.8 or a higher version installed.

2. Clone/download this repository.

3. In the terminal, navigate to the FastSD CPU directory.

4. Run the following commands:
    - `chmod +x install.sh`
    - `./install.sh`

5. To start FastSD CPU, run:
    - `chmod +x start.sh`
    - `./start.sh`.

Feel free to use this improved version while keeping the original text formats intact.
