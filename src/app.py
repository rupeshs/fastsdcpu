from app_settings import AppSettings
from utils import show_system_info
from PIL import Image
from backend.models.lcmdiffusion_setting import DiffusionTask
from frontend.webui.image_variations_ui import generate_image_variations
import constants
import time
from argparse import ArgumentParser

from constants import APP_VERSION, LCM_DEFAULT_MODEL_OPENVINO
from models.interface_types import InterfaceType
from constants import DEVICE
from state import get_settings, get_context

parser = ArgumentParser(description=f"FAST SD CPU {constants.APP_VERSION}")
parser.add_argument(
    "-s",
    "--share",
    action="store_true",
    help="Create sharable link(Web UI)",
    required=False,
)
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument(
    "-g",
    "--gui",
    action="store_true",
    help="Start desktop GUI",
)
group.add_argument(
    "-w",
    "--webui",
    action="store_true",
    help="Start Web UI",
)
group.add_argument(
    "-r",
    "--realtime",
    action="store_true",
    help="Start realtime inference UI(experimental)",
)
group.add_argument(
    "-v",
    "--version",
    action="store_true",
    help="Version",
)
parser.add_argument(
    "--lcm_model_id",
    type=str,
    help="Model ID or path,Default SimianLuo/LCM_Dreamshaper_v7",
    default="SimianLuo/LCM_Dreamshaper_v7",
)
parser.add_argument(
    "--prompt",
    type=str,
    help="Describe the image you want to generate",
    default="",
)
parser.add_argument(
    "--image_height",
    type=int,
    help="Height of the image",
    default=512,
)
parser.add_argument(
    "--image_width",
    type=int,
    help="Width of the image",
    default=512,
)
parser.add_argument(
    "--inference_steps",
    type=int,
    help="Number of steps,default : 4",
    default=4,
)
parser.add_argument(
    "--guidance_scale",
    type=int,
    help="Guidance scale,default : 1.0",
    default=1.0,
)

parser.add_argument(
    "--number_of_images",
    type=int,
    help="Number of images to generate ,default : 1",
    default=1,
)
parser.add_argument(
    "--seed",
    type=int,
    help="Seed,default : -1 (disabled) ",
    default=-1,
)
parser.add_argument(
    "--use_openvino",
    action="store_true",
    help="Use OpenVINO model",
)

parser.add_argument(
    "--use_offline_model",
    action="store_true",
    help="Use offline model",
)
parser.add_argument(
    "--use_safety_checker",
    action="store_true",
    help="Use safety checker",
)
parser.add_argument(
    "--use_lcm_lora",
    action="store_true",
    help="Use LCM-LoRA",
)
parser.add_argument(
    "--base_model_id",
    type=str,
    help="LCM LoRA base model ID,Default Lykon/dreamshaper-8",
    default="Lykon/dreamshaper-8",
)
parser.add_argument(
    "--lcm_lora_id",
    type=str,
    help="LCM LoRA model ID,Default latent-consistency/lcm-lora-sdv1-5",
    default="latent-consistency/lcm-lora-sdv1-5",
)
parser.add_argument(
    "-i",
    "--interactive",
    action="store_true",
    help="Interactive CLI mode",
)
parser.add_argument(
    "-t",
    "--use_tiny_auto_encoder",
    action="store_true",
    help="Use tiny auto encoder for SD (TAESD)",
)
parser.add_argument(
    "-f",
    "--file",
    type=str,
    help="Input image for img2img mode",
    default="",
)
parser.add_argument(
    "--img2img",
    action="store_true",
    help="img2img mode; requires input file via -f argument",
)
parser.add_argument(
    "--batch_count",
    type=int,
    help="Number of sequential generations",
    default=1,
)
parser.add_argument(
    "--strength",
    type=float,
    help="img2img strength",
    default=0.3,
)
parser.add_argument(
    "--upscale",
    action="store_true",
    help="Tiled SD upscale",
)
args = parser.parse_args()

if args.version:
    print(APP_VERSION)
    exit()

# parser.print_help()
show_system_info()
print(f"Using device : {constants.DEVICE}")
if args.webui:
    app_settings = get_settings()
else:
    app_settings = get_settings()

print(f"Found {len(app_settings.lcm_models)} LCM models in config/lcm-models.txt")
print(
    f"Found {len(app_settings.stable_diffsuion_models)} stable diffusion models in config/stable-diffusion-models.txt"
)
print(
    f"Found {len(app_settings.lcm_lora_models)} LCM-LoRA models in config/lcm-lora-models.txt"
)
print(
    f"Found {len(app_settings.openvino_lcm_models)} OpenVINO LCM models in config/openvino-lcm-models.txt"
)
if args.gui:
    from frontend.gui.ui import start_gui

    print("Starting desktop GUI mode(Qt)")
    start_gui(
        [],
        app_settings,
    )
elif args.webui:
    from frontend.webui.ui import start_webui

    print("Starting web UI mode")
    start_webui(
        args.share,
    )
elif args.realtime:
    from frontend.webui.realtime_ui import start_realtime_text_to_image

    print("Starting realtime text to image(EXPERIMENTAL)")
    start_realtime_text_to_image(args.share)
else:
    context = get_context(InterfaceType.CLI)
    config = app_settings.settings

    if args.use_openvino:
        config.lcm_diffusion_setting.lcm_model_id = LCM_DEFAULT_MODEL_OPENVINO
    else:
        config.lcm_diffusion_setting.lcm_model_id = args.lcm_model_id

    config.lcm_diffusion_setting.prompt = args.prompt
    config.lcm_diffusion_setting.image_height = args.image_height
    config.lcm_diffusion_setting.image_width = args.image_width
    config.lcm_diffusion_setting.guidance_scale = args.guidance_scale
    config.lcm_diffusion_setting.number_of_images = args.number_of_images
    config.lcm_diffusion_setting.inference_steps = args.inference_steps
    config.lcm_diffusion_setting.seed = args.seed
    config.lcm_diffusion_setting.use_openvino = args.use_openvino
    config.lcm_diffusion_setting.use_tiny_auto_encoder = args.use_tiny_auto_encoder
    config.lcm_diffusion_setting.use_lcm_lora = args.use_lcm_lora
    config.lcm_diffusion_setting.lcm_lora.base_model_id = args.base_model_id
    config.lcm_diffusion_setting.lcm_lora.lcm_lora_id = args.lcm_lora_id
    config.lcm_diffusion_setting.diffusion_task = DiffusionTask.text_to_image.value

    if args.img2img and args.file != "" :
        config.lcm_diffusion_setting.init_image = Image.open(args.file)
        config.lcm_diffusion_setting.diffusion_task = DiffusionTask.image_to_image.value
    elif args.img2img and args.file == "":
        print("You need to specify a file in img2img mode")
        exit()
    elif args.upscale and args.file == "":
        print("You need to specify a file in SD upscale mode")
        exit()

    if args.seed > -1:
        config.lcm_diffusion_setting.use_seed = True
    else:
        config.lcm_diffusion_setting.use_seed = False
    config.lcm_diffusion_setting.use_offline_model = args.use_offline_model
    config.lcm_diffusion_setting.use_safety_checker = args.use_safety_checker

    if args.interactive:
        while True:
            user_input = input(">>")
            if user_input == "exit":
                break
            config.lcm_diffusion_setting.prompt = user_input
            context.generate_text_to_image(
                settings=config,
                device=DEVICE,
            )

    # Perform Tiled SD upscale
    elif args.upscale:
        input = Image.open(args.file)
        mask = Image.open("configs/mask.png")
        result = Image.new(mode = "RGBA", size = (input.size[0] * 2, input.size[1] * 2), color = (0, 0, 0, 0))

        args.batch_count = 1
        config.lcm_diffusion_setting.image_width = 512
        config.lcm_diffusion_setting.image_height = 512
        config.lcm_diffusion_setting.number_of_images = 1

        total_cols = int(input.size[0] / 256)      # Image width / tile size
        total_rows = int(input.size[1] / 256)      # Image height / tile size
        for y in range(0, total_rows):
            y_offset = y * 16
            for x in range(0, total_cols):
                x_offset = x * 16
                x1 = x * 256 - x_offset
                y1 = y * 256 - y_offset
                x2 = x1 + 256
                y2 = y1 + 256
                config.lcm_diffusion_setting.init_image = input.crop((x1, y1, x2, y2))
                output_tile = generate_image_variations(config.lcm_diffusion_setting.init_image, args.strength)[0]
                result.paste(output_tile, (x * 512 - x_offset * 2, y * 512 - y_offset * 2), mask)
                output_tile.close()
                config.lcm_diffusion_setting.init_image.close()
        result.save("results/fastSD-" + str(int(time.time())) + ".png")
        exit()
    # If img2img argument is set and prompt is empty, use image variations mode
    elif args.img2img and args.prompt == "":
        for i in range(0, args.batch_count):
            generate_image_variations(config.lcm_diffusion_setting.init_image, args.strength)
    else:
        for i in range(0, args.batch_count):
            context.generate_text_to_image(
                settings=config,
                device=DEVICE,
            )
