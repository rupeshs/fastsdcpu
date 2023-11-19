from os import environ

APP_VERSION = "v1.0.0 beta 16"
LCM_DEFAULT_MODEL = "SimianLuo/LCM_Dreamshaper_v7"
LCM_DEFAULT_MODEL_OPENVINO = "rupeshs/LCM-dreamshaper-v7-openvino"
APP_NAME = "FastSD CPU"
APP_SETTINGS_FILE = "settings.yaml"
RESULTS_DIRECTORY = "results"
CONFIG_DIRECTORY = "configs"
DEVICE = environ.get("DEVICE", "cpu")
SD_MODELS_FILE = "stable-diffusion-models.txt"
LCM_LORA_MODELS_FILE = "lcm-lora-models.txt"
OPENVINO_LCM_MODELS_FILE = "openvino-lcm-models.txt"
TAESD_MODEL = "madebyollin/taesd"
TAESDXL_MODEL = "madebyollin/taesdxl"
TAESD_MODEL_OPENVINO = "deinferno/taesd-openvino"
