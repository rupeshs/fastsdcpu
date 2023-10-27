# FastSD CPU: A Quick and Efficient Image Generator! :rocket:

FastSD CPU is a high-speed image generation tool designed for your Core i7-12700 CPU. It creates stunning 512x512 images swiftly and efficiently, taking just 10 seconds per image (thanks to OpenVINO optimization!). This tool is based on the fantastic work done by the creators of [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model).

![FastSD CPU Screenshot](https://raw.githubusercontent.com/rupeshs/fastsdcpu/main/fastsdcpu-screenshot.png)

## Key Features :star2:
- **Various Image Sizes:** FastSD CPU supports image sizes of 256x256, 512x512, and 768x768, allowing you to generate the perfect image for your needs.
- **Cross-Platform Compatibility:** Whether you're on Windows or Linux, FastSD CPU is here for you.
- **Save Your Creations:** FastSD CPU enables you to save your generated images for later use or sharing.
- **Customizable Settings:** You have control over the number of steps, guidance, seed, and safety checks, ensuring you get the output you desire.
- **Enhanced Performance:** With OpenVINO support, experience a whopping 50% speed boost, making your image generation process even faster and more efficient.

## OpenVINO Optimization :zap:

We'd like to extend our gratitude to [deinferno](https://github.com/deinferno) for contributing the OpenVINO model. By default, OpenVINO support is disabled, but if you're on Windows, you can enable it for blazing-fast performance. With OpenVINO, generating a single 512x512 image takes just 10 seconds on a Core i7-12700 processor.

## Supported LCM Models :paintbrush:
FastSD CPU currently supports the LCM model (Dreamshaper_v7) in Diffuser format. You can find these models at the following locations:
- [SimianLuo Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)
- [deinferno Dreamshaper_v7 (OpenVINO optimized)](https://huggingface.co/deinferno/LCM_Dreamshaper_v7-openvino)

## Installation Guide :computer:

### For Windows Users:

1. **Prerequisites:** Ensure you have a working Python installation (Recommended: Python 3.10 or 3.11).

2. **Download and Setup:**
   - Clone/download this repository or grab the latest release.
   - Double click on `install.bat` and wait for the installation to complete (duration depends on your internet speed).

3. **Run FastSD CPU:**
   - Double click `start.bat` to launch FastSD CPU and begin your creative journey!

### For Linux Users:

1. **Python Installation:** Make sure you have Python 3.8 or a higher version installed on your system.

2. **Download and Setup:**
   - Clone/download this repository to your local machine.
   - Navigate to the FastSD CPU directory in your terminal.
   - Run the following commands:
     ```
     chmod +x install.sh
     ./install.sh
     ```

3. **Start Generating:**
   - To start FastSD CPU, use the following commands:
     ```
     chmod +x start.sh
     ./start.sh
     ```

Now you're all set to create amazing images with FastSD CPU! Enjoy the speed and simplicity of your new image generation tool. Happy creating! 🎨✨

### License

The fastsdcpu project is available as open source under the terms of the [MIT license](https://github.com/rupeshs/fastsdcpu/blob/main/LICENSE)
