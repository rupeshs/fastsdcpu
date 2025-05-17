import gradio as gr


def show_error(message: str) -> None:
    if "num_inference_steps != 2" in message:
        gr.Warning(
            "Please set the generation setting inference steps to 2 for SANA sprint model"
        )
    else:
        gr.Warning(message)
