from utils import show_system_info
from PIL import Image
from backend.models.lcmdiffusion_setting import DiffusionTask
from frontend.webui.image_variations_ui import generate_image_variations
from backend.upscale.tiled_upscale import generate_upscaled_image
import constants
import json
from argparse import ArgumentParser
from paths import FastStableDiffusionPaths

from constants import APP_VERSION, LCM_DEFAULT_MODEL_OPENVINO
from models.interface_types import InterfaceType
from constants import DEVICE
from state import get_settings, get_context
from backend.upscale.upscaler import upscale_image

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
    help="Number of steps,default : 1",
    default=1,
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
    help="Denoising strength for img2img and Image variations",
    default=0.3,
)
parser.add_argument(
    "--sdupscale",
    action="store_true",
    help="Tiled SD upscale,works only for the resolution 512x512,scale upto 2X (no OpenVINO support)",
)
parser.add_argument(
    "--upscale",
    action="store_true",
    help="EDSR SD upscale ",
)
parser.add_argument(
    "--custom_settings",
    type=str,
    help="JSON file containing custom generation settings",
    default=None,
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
    config.lcm_diffusion_setting.strength = args.strength
    config.lcm_diffusion_setting.seed = args.seed
    config.lcm_diffusion_setting.use_openvino = args.use_openvino
    config.lcm_diffusion_setting.use_tiny_auto_encoder = args.use_tiny_auto_encoder
    config.lcm_diffusion_setting.use_lcm_lora = args.use_lcm_lora
    config.lcm_diffusion_setting.lcm_lora.base_model_id = args.base_model_id
    config.lcm_diffusion_setting.lcm_lora.lcm_lora_id = args.lcm_lora_id
    config.lcm_diffusion_setting.diffusion_task = DiffusionTask.text_to_image.value

    if args.img2img and args.file != "":
        config.lcm_diffusion_setting.init_image = Image.open(args.file)
        config.lcm_diffusion_setting.diffusion_task = DiffusionTask.image_to_image.value
    elif args.img2img and args.file == "":
        print("Error : You need to specify a file in img2img mode")
        exit()
    elif args.upscale and args.file == "" and args.custom_settings == None:
        print("Error : You need to specify a file in SD upscale mode")
        exit()
    elif args.prompt == "" and args.file == "" and args.custom_settings == None:
        print("Error : You need to provide a prompt")
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

    elif args.upscale:
        image = Image.open(args.file)
        output_path = FastStableDiffusionPaths.get_upscale_filepath(
            args.file,
            2,
            "png",
        )
        result = upscale_image(
            context,
            image,
            output_path,
            2,
        )
    # Perform Tiled SD upscale (EXPERIMENTAL)
    elif args.sdupscale:
        config.lcm_diffusion_setting.strength = 0.3 if args.use_openvino else 0.1
        upscale_settings = None
        output_path = FastStableDiffusionPaths.get_upscale_filepath(
            args.file,
            2,
            "png",
        )
        if args.custom_settings:
            with open(args.custom_settings) as f:
                upscale_settings = json.load(f)

        generate_upscaled_image(
            config,
            Image.open(args.file),
            config.lcm_diffusion_setting.strength,
            upscale_settings=upscale_settings,
            context=context,
            tile_overlap=32 if config.lcm_diffusion_setting.use_openvino else 16,
            output_path=output_path,
        )
        exit()
    # If img2img argument is set and prompt is empty, use image variations mode
    elif args.img2img and args.prompt == "":
        for i in range(0, args.batch_count):
            generate_image_variations(
                config.lcm_diffusion_setting.init_image, args.strength
            )
    else:
        for i in range(0, args.batch_count):
            context.generate_text_to_image(
                settings=config,
                device=DEVICE,
            )
