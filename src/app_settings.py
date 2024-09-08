from copy import deepcopy
from os import makedirs, path

import yaml
from constants import (
    LCM_LORA_MODELS_FILE,
    LCM_MODELS_FILE,
    OPENVINO_LCM_MODELS_FILE,
    SD_MODELS_FILE,
)
from paths import FastStableDiffusionPaths, join_paths
from utils import get_files_in_dir, get_models_from_text_file

from models.settings import Settings


class AppSettings:
    def __init__(self):
        self.config_path = FastStableDiffusionPaths().get_app_settings_path()
        self._stable_diffsuion_models = get_models_from_text_file(
            FastStableDiffusionPaths().get_models_config_path(SD_MODELS_FILE)
        )
        self._lcm_lora_models = get_models_from_text_file(
            FastStableDiffusionPaths().get_models_config_path(LCM_LORA_MODELS_FILE)
        )
        self._openvino_lcm_models = get_models_from_text_file(
            FastStableDiffusionPaths().get_models_config_path(OPENVINO_LCM_MODELS_FILE)
        )
        self._lcm_models = get_models_from_text_file(
            FastStableDiffusionPaths().get_models_config_path(LCM_MODELS_FILE)
        )
        self._gguf_diffusion_models = get_files_in_dir(
            join_paths(FastStableDiffusionPaths().get_gguf_models_path(), "diffusion")
        )
        self._gguf_clip_models = get_files_in_dir(
            join_paths(FastStableDiffusionPaths().get_gguf_models_path(), "clip")
        )
        self._gguf_vae_models = get_files_in_dir(
            join_paths(FastStableDiffusionPaths().get_gguf_models_path(), "vae")
        )
        self._gguf_t5xxl_models = get_files_in_dir(
            join_paths(FastStableDiffusionPaths().get_gguf_models_path(), "t5xxl")
        )
        self._config = None

    @property
    def settings(self):
        return self._config

    @property
    def stable_diffsuion_models(self):
        return self._stable_diffsuion_models

    @property
    def openvino_lcm_models(self):
        return self._openvino_lcm_models

    @property
    def lcm_models(self):
        return self._lcm_models

    @property
    def lcm_lora_models(self):
        return self._lcm_lora_models

    @property
    def gguf_diffusion_models(self):
        return self._gguf_diffusion_models

    @property
    def gguf_clip_models(self):
        return self._gguf_clip_models

    @property
    def gguf_vae_models(self):
        return self._gguf_vae_models

    @property
    def gguf_t5xxl_models(self):
        return self._gguf_t5xxl_models

    def load(self, skip_file=False):
        if skip_file:
            print("Skipping config file")
            settings_dict = self._load_default()
            self._config = Settings.model_validate(settings_dict)
        else:
            if not path.exists(self.config_path):
                base_dir = path.dirname(self.config_path)
                if not path.exists(base_dir):
                    makedirs(base_dir)
                try:
                    print("Settings not found creating default settings")
                    with open(self.config_path, "w") as file:
                        yaml.dump(
                            self._load_default(),
                            file,
                        )
                except Exception as ex:
                    print(f"Error in creating settings : {ex}")
                    exit()
            try:
                with open(self.config_path) as file:
                    settings_dict = yaml.safe_load(file)
                    self._config = Settings.model_validate(settings_dict)
            except Exception as ex:
                print(f"Error in loading settings : {ex}")

    def save(self):
        try:
            with open(self.config_path, "w") as file:
                tmp_cfg = deepcopy(self._config)
                tmp_cfg.lcm_diffusion_setting.init_image = None
                configurations = tmp_cfg.model_dump(
                    exclude=["init_image"],
                )
                if configurations:
                    yaml.dump(configurations, file)
        except Exception as ex:
            print(f"Error in saving settings : {ex}")

    def _load_default(self) -> dict:
        default_config = Settings()
        return default_config.model_dump()
