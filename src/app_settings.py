import yaml
from os import path, makedirs
from models.settings import Settings
from paths import FastStableDiffusionPaths


class AppSettings:
    def __init__(self):
        self.config_path = FastStableDiffusionPaths().get_app_settings_path()

    @property
    def settings(self):
        return self._config

    def load(self):
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
                self._config = Settings.parse_obj(settings_dict)
        except Exception as ex:
            print(f"Error in loading settings : {ex}")

    def save(self):
        try:
            with open(self.config_path, "w") as file:
                yaml.dump(self._config.dict(), file)
        except Exception as ex:
            print(f"Error in saving settings : {ex}")

    def _load_default(self) -> dict:
        defult_config = Settings()
        return defult_config.dict()
