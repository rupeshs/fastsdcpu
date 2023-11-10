import os
import constants


def join_paths(
    first_path: str,
    second_path: str,
) -> str:
    return os.path.join(first_path, second_path)


def get_app_path():
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
    def get_css_path():
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
    def get_stable_diffusion_models_config_path():
        configs_path = get_configs_path()
        models_path = join_paths(
            configs_path,
            constants.SD_MODELS_FILE,
        )
        return models_path

    @staticmethod
    def get_lcm_lora_models_config_path():
        configs_path = get_configs_path()
        lcm_lora_models_path = join_paths(
            configs_path,
            constants.LCM_LORA_MODELS_FILE,
        )
        return lcm_lora_models_path
