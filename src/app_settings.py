import yaml
from os import path, makedirs
from models.settings import Settings
from paths import FastStableDiffusionPaths


class AppSettings:
    def __init__(self):
        self.config_path = FastStableDiffusionPaths().get_app_settings_path()

    def load(self):
        if not path.exists(self.config_path):
            base_dir = path.dirname(self.config_path)
            if not path.exists(base_dir):
                makedirs(base_dir)
            try:
                with open(self.config_path, "w") as file:
                    yaml.dump(
                        self.load_default(),
                        file,
                    )
            except Exception as ex:
                print(f"Error in creating settings : {ex}")
        try:
            with open(self.config_path) as file:
                settings_dict = yaml.safe_load(file)
                self._config = Settings.parse_obj(settings_dict)
        except Exception as ex:
            print(f"Error in loading settings : {ex}")

    def get_settings(self) -> Settings:
        return self._config

    def save(self, settings: Settings):
        print(dict(settings))
        try:
            with open(self.config_path, "w") as file:
                yaml.dump(dict(settings), file)
        except Exception as ex:
            print(f"Error in saving settings : {ex}")

    def load_default(self) -> dict:
        defult_config = Settings()
        return dict(defult_config)
