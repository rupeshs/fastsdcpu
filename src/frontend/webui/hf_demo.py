import base64
from datetime import datetime
from pprint import pprint
from time import perf_counter

import gradio as gr
import numpy as np
from cv2 import imencode
from PIL import Image
from transformers import pipeline

from backend.device import get_device_name, is_openvino_device
from backend.lcm_text_to_image import LCMTextToImage
from backend.models.lcmdiffusion_setting import (
    DiffusionTask,
    LCMDiffusionSetting,
    LCMLora,
)
from backend.safety_checker import SafetyChecker
from constants import APP_VERSION, DEVICE, LCM_DEFAULT_MODEL_OPENVINO

lcm_text_to_image = LCMTextToImage()
lcm_lora = LCMLora(
    base_model_id="Lykon/dreamshaper-7",
    lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
)
classifier = pipeline(
    "image-classification",
    model="Falconsai/nsfw_image_detection",
)
safety_checker = SafetyChecker()


# https://github.com/gradio-app/gradio/issues/2635#issuecomment-1423531319
def encode_pil_to_base64_new(pil_image):
    image_arr = np.asarray(pil_image)[:, :, ::-1]
    _, byte_data = imencode(".png", image_arr)
    base64_data = base64.b64encode(byte_data)
    base64_string_opencv = base64_data.decode("utf-8")
    return "data:image/png;base64," + base64_string_opencv


# monkey patching encode pil
gr.processing_utils.encode_pil_to_base64 = encode_pil_to_base64_new


def predict(
    prompt,
    steps,
    seed,
    use_seed,
):
    print(f"prompt - {prompt}")
    lcm_diffusion_setting = LCMDiffusionSetting()
    lcm_diffusion_setting.lcm_model_id = "rupeshs/hyper-sd-sdxl-1-step"
    lcm_diffusion_setting.diffusion_task = DiffusionTask.text_to_image.value
    lcm_diffusion_setting.openvino_lcm_model_id = "rupeshs/sd-turbo-openvino"
    lcm_diffusion_setting.use_lcm_lora = False
    lcm_diffusion_setting.prompt = prompt
    lcm_diffusion_setting.guidance_scale = 1.0
    lcm_diffusion_setting.inference_steps = steps
    lcm_diffusion_setting.seed = seed
    lcm_diffusion_setting.use_seed = use_seed
    lcm_diffusion_setting.use_safety_checker = True
    lcm_diffusion_setting.use_tiny_auto_encoder = False
    # lcm_diffusion_setting.image_width = 320 if is_openvino_device() else 512
    # lcm_diffusion_setting.image_height = 320 if is_openvino_device() else 512
    lcm_diffusion_setting.image_width = 512
    lcm_diffusion_setting.image_height = 512
    lcm_diffusion_setting.use_openvino = True
    lcm_diffusion_setting.use_tiny_auto_encoder = True
    pprint(lcm_diffusion_setting.model_dump())
    lcm_text_to_image.init(lcm_diffusion_setting=lcm_diffusion_setting)
    start = perf_counter()
    images = lcm_text_to_image.generate(lcm_diffusion_setting)
    latency = perf_counter() - start
    print(f"Latency: {latency:.2f} seconds")
    result = images[0]
    if safety_checker.is_safe(
        result,
    ):
        return result  # .resize([512, 512], Image.LANCZOS)
    else:
        print("Unsafe image detected")
        return Image.new("RGB", (512, 512), (0, 0, 0))


css = """
#container{
    margin: 0 auto;
    max-width: 40rem;
}
#intro{
    max-width: 100%;
    text-align: center;
    margin: 0 auto;
}
#generate_button {
    color: white;
    border-color: #007bff;
    background: #007bff;
    width: 200px;
    height: 50px;
}
footer {
    visibility: hidden
}
"""


def _get_footer_message() -> str:
    version = f"<center><p> {APP_VERSION} "
    current_year = datetime.now().year
    footer_msg = version + (
        f'  © {current_year} <a href="https://github.com/rupeshs">'
        " Rupesh Sreeraman</a></p></center>"
    )
    warning_msg = "<p><b> Please note that this is a minimal demo app.</b> </p><br>"
    return warning_msg + footer_msg


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="container"):
        use_openvino = "" if is_openvino_device() else ""
        gr.Markdown(
            f"""# FastSD CPU demo {use_openvino}
               **Device : {DEVICE.upper()} , {get_device_name()} | OpenVINO**
            """,
            elem_id="intro",
        )
        gr.HTML(
            f"""
            <p id="project-links" align="center">
                <a href='https://github.com/rupeshs/fastsdcpu'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
            </p> 
                    """
        )

        with gr.Row():
            with gr.Row():
                prompt = gr.Textbox(
                    placeholder="Describe the image you'd like to see",
                    scale=5,
                    container=False,
                )
                generate_btn = gr.Button(
                    "Generate",
                    scale=1,
                    elem_id="generate_button",
                )

        image = gr.Image(type="filepath")
        with gr.Accordion("Advanced options", open=False):
            steps = gr.Slider(
                label="Steps",
                value=1,
                minimum=1,
                maximum=3,
                step=1,
            )
            seed = gr.Slider(
                randomize=True,
                minimum=0,
                maximum=999999999,
                label="Seed",
                step=1,
            )
            seed_checkbox = gr.Checkbox(
                label="Use seed",
                value=False,
                interactive=True,
            )
        gr.HTML(_get_footer_message())

        inputs = [prompt, steps, seed, seed_checkbox]
        generate_btn.click(fn=predict, inputs=inputs, outputs=image)


def start_demo():
    demo.queue()
    demo.launch(share=False)
