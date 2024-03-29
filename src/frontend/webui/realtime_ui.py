import gradio as gr
from backend.lcm_text_to_image import LCMTextToImage
from backend.models.lcmdiffusion_setting import LCMLora, LCMDiffusionSetting
from constants import DEVICE, LCM_DEFAULT_MODEL_OPENVINO
from time import perf_counter
import numpy as np
from cv2 import imencode
import base64
from backend.device import get_device_name
from constants import APP_VERSION
from backend.device import is_openvino_device

lcm_text_to_image = LCMTextToImage()
lcm_lora = LCMLora(
    base_model_id="Lykon/dreamshaper-8",
    lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
)


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
):
    lcm_diffusion_setting = LCMDiffusionSetting()
    lcm_diffusion_setting.openvino_lcm_model_id = "rupeshs/sdxs-512-0.9-openvino"
    lcm_diffusion_setting.prompt = prompt
    lcm_diffusion_setting.guidance_scale = 1.0
    lcm_diffusion_setting.inference_steps = steps
    lcm_diffusion_setting.seed = seed
    lcm_diffusion_setting.use_seed = True
    lcm_diffusion_setting.image_width = 512
    lcm_diffusion_setting.image_height = 512
    lcm_diffusion_setting.use_openvino = True if is_openvino_device() else False
    lcm_diffusion_setting.use_tiny_auto_encoder = True
    lcm_text_to_image.init(
        DEVICE,
        lcm_diffusion_setting,
    )
    start = perf_counter()

    images = lcm_text_to_image.generate(lcm_diffusion_setting)
    latency = perf_counter() - start
    print(f"Latency: {latency:.2f} seconds")
    return images[0]


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
    footer_msg = version + (
        '  © 2023 - 2024 <a href="https://github.com/rupeshs">'
        " Rupesh Sreeraman</a></p></center>"
    )
    return footer_msg


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="container"):
        use_openvino = "- OpenVINO" if is_openvino_device() else ""
        gr.Markdown(
            f"""# Realtime FastSD CPU {use_openvino}
               **Device : {DEVICE} , {get_device_name()}**
            """,
            elem_id="intro",
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

        steps = gr.Slider(
            label="Steps",
            value=1,
            minimum=1,
            maximum=6,
            step=1,
            visible=False,
        )
        seed = gr.Slider(
            randomize=True,
            minimum=0,
            maximum=999999999,
            label="Seed",
            step=1,
        )
        gr.HTML(_get_footer_message())

        inputs = [prompt, steps, seed]
        prompt.input(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        generate_btn.click(
            fn=predict, inputs=inputs, outputs=image, show_progress=False
        )
        steps.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        seed.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)


def start_realtime_text_to_image(share=False):
    demo.queue()
    demo.launch(share=share)
