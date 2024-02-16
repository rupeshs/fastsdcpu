import curses
from PIL import Image
from typing import Any

from constants import DEVICE
from paths import FastStableDiffusionPaths
from backend.upscale.upscaler import upscale_image
from backend.models.lcmdiffusion_setting import DiffusionTask
from backend.upscale.tiled_upscale import generate_upscaled_image
from frontend.webui.image_variations_ui import generate_image_variations

_batch_count = 1



def interactive_mode(config, context):
    print("=============================================")
    print("Welcome to FastSD CPU Interactive CLI")
    print("=============================================")
    print("> 1. Text to Image")
    print("> 2. Image to Image")
    print("> 3. Image Variations")
    print("> 4. EDSR Upscale")
    print("> 5. SD Upscale")
    option = int(input("Enter a Diffusion Task number: "))
    if option not in range(0, 6):
        print("Wrong Diffusion Task number!")
        exit()

    edit_settings = input("Edit default generation settings? (y/N): ")
    if (edit_settings.upper() == 'Y'):
        interactive_settings(config, context)

    if option == 1:
        interactive_txt2img(config, context)
    elif option == 2:
        interactive_img2img(config, context)
    elif option == 3:
        interactive_variations(config, context)
    elif option == 4:
        interactive_edsr(config, context)
    elif option == 5:
        interactive_sdupscale(config, context)



def user_value(value_type: type, message: str, default_value: Any) -> Any:
    try:
        value = value_type(input(message))
    except:
        value = default_value;
    return value



def interactive_settings(config, context):
    global _batch_count
    c = config.lcm_diffusion_setting
    print("Enter generation settings (leave empty to use current value)")
    c.image_width = user_value(int, f"Image width ({c.image_width}): ", c.image_width)
    c.image_height = user_value(int, f"Image height ({c.image_height}): ", c.image_height)
    c.inference_steps = user_value(int, f"Inference steps ({c.inference_steps}): ", c.inference_steps)
    c.number_of_images = user_value(int, f"Number of images per batch ({c.number_of_images}): ", c.number_of_images)
    _batch_count = user_value(int, f"Batch count ({_batch_count}): ", _batch_count)
    # output_format = user_value(int, f"Output format (PNG)", 1)
    print(config.lcm_diffusion_setting)



def interactive_txt2img(config, context):
    global _batch_count
    config.lcm_diffusion_setting.diffusion_task = DiffusionTask.text_to_image.value
    user_input = input("Write a prompt (write 'exit' to quit): ")
    while True:
        if user_input == "exit":
            exit()
        config.lcm_diffusion_setting.prompt = user_input
        for i in range(0, _batch_count):
            context.generate_text_to_image(
                settings=config,
                device=DEVICE,
            )
        user_input = input("Write a prompt: ")



def interactive_img2img(config, context):
    global _batch_count
    c = config.lcm_diffusion_setting
    c.diffusion_task = DiffusionTask.image_to_image.value
    source_path = input("Image path: ")
    if source_path == "":
        print("Error : You need to provide a file in img2img mode")
        exit()
    c.strength = user_value(float, f"img2img strength ({c.strength}): ", c.strength)
    user_input = input("Write a prompt (write 'exit' to quit): ")
    while True:
        if user_input == "exit":
            exit()
        c.init_image = Image.open(source_path)
        c.prompt = user_input
        for i in range(0, _batch_count):
            context.generate_text_to_image(
                settings=config,
                device=DEVICE,
            )
        new_path = input(f"Image path ({source_path}): ")
        if new_path != "":
            source_path = new_path
        c.strength = user_value(float, f"img2img strength ({c.strength}): ", c.strength)
        user_input = input("Write a prompt: ")



def interactive_variations(config, context):
    global _batch_count
    c = config.lcm_diffusion_setting
    c.diffusion_task = DiffusionTask.image_to_image.value
    source_path = input("Image path: ")
    if source_path == "":
        print("Error : You need to provide a file in Image variations mode")
        exit()
    c.strength = user_value(float, f"Image variations strength ({c.strength}): ", c.strength)
    while True:
        c.init_image = Image.open(source_path)
        c.prompt = ""
        for i in range(0, _batch_count):
            generate_image_variations(
                c.init_image, c.strength
            )
        user_input = input("Continue in Image variations mode? (Y/n): ")
        if user_input.upper() == "N":
            exit()
        new_path = input(f"Image path ({source_path}): ")
        if new_path != "":
            source_path = new_path
        c.strength = user_value(float, f"Image variations strength ({c.strength}): ", c.strength)



def interactive_edsr(config, context):
    source_path = input("Image path: ")
    if source_path == "":
        print("Error : You need to provide a file in EDSR mode")
        exit()
    while True:
        output_path = FastStableDiffusionPaths.get_upscale_filepath(
            source_path,
            2,
            config.generated_images.format,
        )
        result = upscale_image(
            context,
            source_path,
            output_path,
            2,
        )
        user_input = input("Continue in EDSR upscale mode? (Y/n): ")
        if user_input.upper() == "N":
            exit()
        new_path = input(f"Image path ({source_path}): ")
        if new_path != "":
            source_path = new_path



def interactive_sdupscale(config, context):
    c = config.lcm_diffusion_setting
    c.diffusion_task = DiffusionTask.image_to_image.value
    source_path = input("Image path: ")
    if source_path == "":
        print("Error : You need to provide a file in SD Upscale mode")
        exit()
    c.strength = user_value(float, f"SD Upscale strength ({c.strength}): ", c.strength)
    if c.use_openvino:
        c.strength = 0.3
    while True:
        output_path = FastStableDiffusionPaths.get_upscale_filepath(
            source_path,
            2,
            config.generated_images.format,
        )
        generate_upscaled_image(
            config,
            source_path,
            c.strength,
            upscale_settings=None,
            context=context,
            tile_overlap=32 if c.use_openvino else 16,
            output_path=output_path,
            image_format=config.generated_images.format,
        )
        user_input = input("Continue in SD Upscale mode? (Y/n): ")
        if user_input.upper() == "N":
            exit()
        new_path = input(f"Image path ({source_path}): ")
        if new_path != "":
            source_path = new_path
        c.strength = user_value(float, f"SD Upscale strength ({c.strength}): ", c.strength)


