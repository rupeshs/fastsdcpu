import gradio as gr
from backend.lcm_text_to_image import LCMTextToImage
from backend.models.lcmdiffusion_setting import LCMLora, LCMDiffusionSetting

lcm_text_to_image = LCMTextToImage()
lcm_lora = LCMLora(
    base_model_id="Lykon/dreamshaper-7",
    lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
)


def predict(
    prompt,
    guidance,
    steps,
    seed=1231231,
):
    lcm_text_to_image.init(
        model_id="",
        use_lora=True,
        lcm_lora=lcm_lora,
    )
    lcm_diffusion_setting = LCMDiffusionSetting()
    lcm_diffusion_setting.prompt = prompt
    lcm_diffusion_setting.guidance_scale = guidance
    lcm_diffusion_setting.inference_steps = steps
    images = lcm_text_to_image.generate(lcm_diffusion_setting)
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
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown(
            """# Realtime text to image generation using LCM-LoRA 
            """,
            elem_id="intro",
        )
        with gr.Row():
            with gr.Row():
                prompt = gr.Textbox(
                    placeholder="Insert your prompt here:", scale=5, container=False
                )
                generate_bt = gr.Button("Generate", scale=1)
        with gr.Accordion("Advanced options", open=False):
            guidance = gr.Slider(
                label="Guidance",
                minimum=1.0,
                maximum=2.0,
                value=1.0,
                step=0.001,
            )
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
                maximum=12013012031030,
                label="Seed",
                step=1,
            )
        image = gr.Image(type="filepath")

        inputs = [prompt, guidance, steps, seed]
        generate_bt.click(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        prompt.input(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        guidance.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        steps.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        seed.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)


def start_realtime_text_to_image():
    demo.queue()
    demo.launch(share=True)
