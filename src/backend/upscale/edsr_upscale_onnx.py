import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from PIL import Image


def upscale_edsr_2x(image_path: str):
    input_image = Image.open(image_path).convert("RGB")
    input_image = np.array(input_image).astype("float32")
    input_image = np.transpose(input_image, (2, 0, 1))
    img_arr = np.expand_dims(input_image, axis=0)

    if np.max(img_arr) > 256:  # 16-bit image
        max_range = 65535
    else:
        max_range = 255.0
        img = img_arr / max_range

    model_path = hf_hub_download(
        repo_id="rupeshs/edsr-onnx",
        filename="edsr_onnxsim_2x.onnx",
    )
    sess = onnxruntime.InferenceSession(model_path)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run(
        [output_name],
        {input_name: img},
    )[0]

    result = output.squeeze()
    result = result.clip(0, 1)
    image_array = np.transpose(result, (1, 2, 0))
    image_array = np.uint8(image_array * 255)
    upscaled_image = Image.fromarray(image_array)
    return upscaled_image
