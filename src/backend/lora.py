import glob
from os import path
from paths import get_file_name, FastStableDiffusionPaths
from pathlib import Path


class _lora_info:
    """
    A basic class to keep track of the currently loaded LoRAs and their weights.

    The diffusers function _get_active_adapters()_ returns a list of adapter
    names but not their weights so we need a way to keep track of the current
    LoRA weights to set whenever a new LoRA is loaded.
    """

    def __init__(
        self,
        path: str,
        weight: float,
    ):
        self.path = path
        self.adapter_name = get_file_name(path)
        self.weight = weight

    def __del__(self):
        self.path = None
        self.adapter_name = None


_loaded_loras = []
_current_pipeline = None


def load_lora_weight(
    pipeline,
    lcm_diffusion_setting,
):
    """
    Loads a LoRA from the LoRA path setting.

    This function loads a LoRA from the LoRA path stored in the settings so
    it's possible to load multiple LoRAs by calling this function more than
    once with a different LoRA path setting; note that if you plan to load
    multiple LoRAs and dynamically change their weights, you might want to
    set the LoRA fuse option to _False_.
    """
    if not lcm_diffusion_setting.lora.path:
        raise Exception("Empty lora model path")

    if not path.exists(lcm_diffusion_setting.lora.path):
        raise Exception("Lora model path is invalid")

    # If the pipeline has been rebuilt since the last call, remove all
    # references to previously loaded LoRAs and store the new pipeline
    global _loaded_loras
    global _current_pipeline
    if pipeline != _current_pipeline:
        reset_active_lora_weights()
        _current_pipeline = pipeline

    current_lora = _lora_info(
        lcm_diffusion_setting.lora.path,
        lcm_diffusion_setting.lora.weight,
    )
    _loaded_loras.append(current_lora)

    if lcm_diffusion_setting.lora.enabled:
        print(f"LoRA adapter name : {current_lora.adapter_name}")
        pipeline.load_lora_weights(
            FastStableDiffusionPaths.get_lora_models_path(),
            weight_name=Path(lcm_diffusion_setting.lora.path).name,
            local_files_only=True,
            adapter_name=current_lora.adapter_name,
        )
        update_lora_weights(
            pipeline,
            lcm_diffusion_setting,
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


def get_active_lora_weights():
    """
    Returns a list of _(adapter_name, weight)_ tuples for the currently loaded LoRAs.
    """
    active_loras = []
    for lora_info in _loaded_loras:
        active_loras.append(
            (
                lora_info.adapter_name,
                lora_info.weight,
            )
        )
    return active_loras


def reset_active_lora_weights():
    """
    Clears the global list of active LoRA weights.

    This method clears the list of active LoRA weights but it doesn't actually
    remove the active LoRA weights from the current generation pipeline.
    This method is only meant to be called when rebuilding the generation pipeline.
    """
    global _loaded_loras
    for lora in _loaded_loras:
        del lora
    del _loaded_loras
    _loaded_loras = []


def update_lora_weights(
    pipeline,
    lcm_diffusion_setting,
    lora_weights=None,
):
    """
    Updates the LoRA weights for the currently active LoRAs.

    Args:
        pipeline: The currently active pipeline.
        lcm_diffusion_setting: The global settings, needed to verify if the
            pipeline is running in LCM-LoRA mode.
        lora_weights: An optional list of updated _(adapter_name, weight)_ tuples.
    """
    global _loaded_loras
    global _current_pipeline
    if pipeline != _current_pipeline:
        print("Wrong pipeline when trying to update LoRA weights")
        return
    if lora_weights:
        for idx, lora in enumerate(lora_weights):
            if _loaded_loras[idx].adapter_name != lora[0]:
                print("Wrong adapter name in LoRA enumeration!")
                continue
            _loaded_loras[idx].weight = lora[1]

    adapter_names = []
    adapter_weights = []
    if lcm_diffusion_setting.use_lcm_lora:
        adapter_names.append("lcm")
        adapter_weights.append(1.0)
    for lora in _loaded_loras:
        adapter_names.append(lora.adapter_name)
        adapter_weights.append(lora.weight)
    pipeline.set_adapters(
        adapter_names,
        adapter_weights=adapter_weights,
    )
    adapter_weights = zip(adapter_names, adapter_weights)
    print(f"Adapters: {list(adapter_weights)}")
