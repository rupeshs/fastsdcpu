from os import environ, cpu_count

cpu_cores = cpu_count()
cpus = cpu_cores // 2 if cpu_cores else 0
APP_VERSION = "v1.0.0 beta 252"
LCM_DEFAULT_MODEL = "stabilityai/sd-turbo"
LCM_DEFAULT_MODEL_OPENVINO = "rupeshs/sd-turbo-openvino"
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
TAESD_MODEL_OPENVINO = "rupeshs/taesd-ov"
LCM_MODELS_FILE = "lcm-models.txt"
TAESDXL_MODEL_OPENVINO = "rupeshs/taesdxl-openvino"
LORA_DIRECTORY = "lora_models"
CONTROLNET_DIRECTORY = "controlnet_models"
MODELS_DIRECTORY = "models"
GGUF_THREADS = environ.get("GGUF_THREADS", cpus)
TAEF1_MODEL_OPENVINO = "rupeshs/taef1-openvino"
SAFETY_CHECKER_MODEL = "Falconsai/nsfw_image_detection"
