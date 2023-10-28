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
