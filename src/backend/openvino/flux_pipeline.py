from pathlib import Path

from constants import DEVICE, LCM_DEFAULT_MODEL_OPENVINO, TAEF1_MODEL_OPENVINO
from huggingface_hub import snapshot_download

from backend.openvino.ovflux import (
    TEXT_ENCODER_2_PATH,
    TEXT_ENCODER_PATH,
    TRANSFORMER_PATH,
    VAE_DECODER_PATH,
    init_pipeline,
)


def get_flux_pipeline(
    model_id: str = LCM_DEFAULT_MODEL_OPENVINO,
    use_taef1: bool = False,
    taef1_path: str = TAEF1_MODEL_OPENVINO,
):
    model_dir = Path(snapshot_download(model_id))
    vae_dir = Path(snapshot_download(taef1_path)) if use_taef1 else model_dir

    model_dict = {
        "transformer": model_dir / TRANSFORMER_PATH,
        "text_encoder": model_dir / TEXT_ENCODER_PATH,
        "text_encoder_2": model_dir / TEXT_ENCODER_2_PATH,
        "vae": vae_dir / VAE_DECODER_PATH,
    }
    ov_pipe = init_pipeline(
        model_dir,
        model_dict,
        device=DEVICE.upper(),
        use_taef1=use_taef1,
    )

    return ov_pipe
