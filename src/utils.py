import platform
from typing import List


def show_system_info():
    try:
        print(f"Running on {platform.system()} platform")
        print(f"OS: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
    except Exception as ex:
        print(f"Error occurred while getting system information {ex}")


def get_models_from_text_file(file_path: str) -> List:
    models = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    for repo_id in lines:
        if repo_id.strip() != "":
            models.append(repo_id.strip())
    return models


def get_image_file_extension(image_format: str) -> str:
    if image_format == "JPEG":
        return ".jpg"
    elif image_format == "PNG":
        return ".png"
