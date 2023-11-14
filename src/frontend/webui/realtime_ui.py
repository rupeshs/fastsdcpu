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

lcm_text_to_image = LCMTextToImage()
lcm_lora = LCMLora(
    base_model_id="Lykon/dreamshaper-7",
    lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
)


# https://github.com/gradio-app/gradio/issues/2635#issuecomment-1423531319
def encode_pil_to_base64_new(pil_image):
    image_arr = np.asarray(pil_image)[:, :, ::-1]
    _, byte_data = imencode(".jpg", image_arr)
    base64_data = base64.b64encode(byte_data)
    base64_string_opencv = base64_data.decode("utf-8")
    return "data:image/png;base64," + base64_string_opencv


# monkey patching encode pil
gr.processing_utils.encode_pil_to_base64 = encode_pil_to_base64_new


def predict(
    prompt,
    steps,
    seed=123123,
):
    lcm_text_to_image.init(
        model_id=LCM_DEFAULT_MODEL_OPENVINO,
        use_lora=True,
        lcm_lora=lcm_lora,
        use_openvino=True if DEVICE == "cpu" else False,
    )

    lcm_diffusion_setting = LCMDiffusionSetting()
    lcm_diffusion_setting.prompt = prompt
    lcm_diffusion_setting.guidance_scale = 1.0
    lcm_diffusion_setting.inference_steps = steps
    lcm_diffusion_setting.seed = seed
    lcm_diffusion_setting.use_seed = True
    lcm_diffusion_setting.image_width = 320 if DEVICE == "cpu" else 512
    lcm_diffusion_setting.image_height = 320 if DEVICE == "cpu" else 512
    lcm_diffusion_setting.use_openvino = True if DEVICE == "cpu" else False
    lcm_diffusion_setting.use_tiny_auto_encoder = True
    start = perf_counter()
    images = lcm_text_to_image.generate(lcm_diffusion_setting)
    print(perf_counter() - start)
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
        '  © 2023 <a href="https://github.com/rupeshs">'
        " Rupesh Sreeraman</a></p></center>"
    )
    return footer_msg


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="container"):
        use_openvino = "OpenVINO" if DEVICE == "cpu" else ""
        gr.Markdown(
            f"""# Realtime FastSD CPU - {use_openvino}
               Device : {DEVICE} , {get_device_name()}
            """,
            elem_id="intro",
        )

        with gr.Row():
            with gr.Row():
                prompt = gr.Textbox(
                    placeholder="Insert your prompt here:",
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
                value=4,
                minimum=1,
                maximum=6,
                step=1,
            )
            seed = gr.Slider(
                randomize=True,
                minimum=0,
                maximum=999999999,
                label="Seed",
                step=1,
                value=12123,
            )
        gr.HTML(_get_footer_message())

        inputs = [prompt, steps, seed]
        prompt.input(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        generate_btn.click(
            fn=predict, inputs=inputs, outputs=image, show_progress=False
        )
        steps.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        seed.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)


def start_realtime_text_to_image():
    demo.queue()
    demo.launch(share=False)
