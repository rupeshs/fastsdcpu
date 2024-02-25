import glob
from os import path

from paths import get_file_name


def load_lora_weight(
    pipeline,
    lcm_diffusion_setting,
):
    if not lcm_diffusion_setting.lora.path:
        raise Exception("Empty lora model path")

    if not path.exists(lcm_diffusion_setting.lora.path):
        raise Exception("Lora model path is invalid")

    if lcm_diffusion_setting.lora.enabled:
        adapter_name = get_file_name(lcm_diffusion_setting.lora.path)
        print(f"LoRA adapter name : {adapter_name}")
        pipeline.load_lora_weights(
            lcm_diffusion_setting.lora.path,
            local_files_only=True,
            adapter_name=adapter_name,
        )
        if lcm_diffusion_setting.use_lcm_lora:
            pipeline.set_adapters(
                [
                    "lcm",
                    adapter_name,
                ],
                adapter_weights=[
                    1.0,
                    lcm_diffusion_setting.lora.weight,
                ],
            )
        else:
            pipeline.set_adapters(
                [
                    adapter_name,
                ],
                adapter_weights=[
                    lcm_diffusion_setting.lora.weight,
                ],
            )

        if lcm_diffusion_setting.lora.fuse:
            pipeline.fuse_lora()


def get_lora_models(root_dir: str):
    lora_models = glob.glob(f"{root_dir}/**/*.safetensors", recursive=True)
    lora_models_map = {}
    for file_path in lora_models:
        lora_name = get_file_name(file_path)
        if lora_name is not None:
            lora_models_map[lora_name] = file_path
    return lora_models_map
