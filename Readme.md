### FastSD CPU :sparkles:

Faster version of stable diffusion running on CPU.

Took 10 seconds to generate single 512x512 image on Core i7-12700(With OpenVINO).

Based on [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model) .

![Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/fastsdcpu-screenshot.png)

## Features
 - Image Sizes: Customize your images with support for 256, 512, and 768 sizes.
 - Platform Compatibility: Seamlessly run the application on both Windows and Linux.
 - Save Your Creations: Easily save your edited images with the built-in save feature.
 - Fine-Tune Settings: Take control of your editing process with adjustable settings for the number of steps, guidance, and seed.

## Safety First:
 - Enhanced Safety Checker: Ensuring your creations are worry-free.

## Maximum Potential:
 - Push the Limits: Maximum inference steps extended to 25.
 - OpenVINO Magic: Exclusive support for OpenVINO - Thanks to deinferno's contribution.
 - Windows Wonder: Enjoy a 50% speed boost, with a single 512x512 image generated in just 10 seconds on a Core i7-12700.

## OpenVINO support

Thanks [deinferno](https://github.com/deinferno) for the OpenVINO model contribution.
Currently, it is tested on Windows only,by default it is disabled.
We found a 50% speed improvement using OpenVINO.It 
Took 10 seconds to generate single 512x512 image on Core i7-12700

## Unlock Your Dreams with LCM Models: 

Step into Dreamshaper_v7's world, supported in Diffuser format.

- Model URLs for the curious:
- https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
- https://huggingface.co/deinferno/LCM_Dreamshaper_v7-openvino

## FastSD CPU on Windows

 - Windows Wonders:
   - A smooth Python experience (Recommended: Python 3.10 or 3.11).
 - Experience the Art:

   - Quick Installation:
       - Clone or download this repository.
       - Execute `install.bat` with patience (installation time depends on your internet speed).

 -  The Magic Begins:
       - Start FastSD CPU by simply double-clicking `start.bat.`

## FastSD CPU on Linux

 - Linux Lovers:
   - Ensure Python 3.8 or higher is on your side.
 - Your Art Studio:
   - Installation Ritual:
       - Clone or download this repository.
       - Navigate to the "fastsdcpu" directory in your terminal.
 - The Linux Magic:
   - Run `install.sh` with `chmod +x install.sh` and `./install.sh.`
   - Start the enchantment by running `start.sh` with `chmod +x start.sh` and `./start.sh.`

## Let Your Imagination Run Wild - Embrace the Magic!
