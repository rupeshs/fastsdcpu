import os
import constants
from pathlib import Path
from time import time
from utils import get_image_file_extension


def join_paths(
    first_path: str,
    second_path: str,
) -> str:
    return os.path.join(first_path, second_path)


def get_file_name(file_path: str) -> str:
    return Path(file_path).stem


def get_app_path() -> str:
    app_dir = os.path.dirname(__file__)
    work_dir = os.path.dirname(app_dir)
    return work_dir


def get_configs_path() -> str:
    config_path = join_paths(get_app_path(), constants.CONFIG_DIRECTORY)
    return config_path


class FastStableDiffusionPaths:
    @staticmethod
    def get_app_settings_path() -> str:
        configs_path = get_configs_path()
        settings_path = join_paths(
            configs_path,
            constants.APP_SETTINGS_FILE,
        )
        return settings_path

    @staticmethod
    def get_results_path() -> str:
        results_path = join_paths(get_app_path(), constants.RESULTS_DIRECTORY)
        return results_path

    @staticmethod
    def get_css_path() -> str:
        app_dir = os.path.dirname(__file__)
        css_path = os.path.join(
            app_dir,
            "frontend",
            "webui",
            "css",
            "style.css",
        )
        return css_path

    @staticmethod
    def get_models_config_path(model_config_file: str) -> str:
        configs_path = get_configs_path()
        models_path = join_paths(
            configs_path,
            model_config_file,
        )
        return models_path

    @staticmethod
    def get_upscale_filepath(
        file_path_src: str,
        scale_factor: int,
        format: str,
    ) -> str:
        if file_path_src:
            file_name_src = get_file_name(file_path_src)
        else:
            file_name_src = "fastsdcpu"

        extension = get_image_file_extension(format)
        upscaled_filepath = join_paths(
            FastStableDiffusionPaths.get_results_path(),
            f"{file_name_src}_{int(scale_factor)}x_upscale_{int(time())}{extension}",
        )
        return upscaled_filepath

    @staticmethod
    def get_lora_models_path() -> str:
        lora_models_path = join_paths(get_app_path(), constants.LORA_DIRECTORY)
        return lora_models_path

    @staticmethod
    def get_controlnet_models_path() -> str:
        controlnet_models_path = join_paths(
            get_app_path(), constants.CONTROLNET_DIRECTORY
        )
        return controlnet_models_path


def get_base_folder_name(path: str) -> str:
    return os.path.basename(path)
