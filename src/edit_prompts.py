import json
from pathlib import Path
from paths import get_configs_path


default_edit_prompts = {
    "restore_old_photo": {
        "display_name": "Restore old photo",
        "prompt": "Full professional restoration of this vintage photograph. Remove all damage including tears, fading, scratches, and discoloration,keep same color tone as original photo",
    },
    "restore_old_photo_colorize": {
        "display_name": "Restore old photo and colorize",
        "prompt": "Full professional restoration of this vintage photograph. Remove all damage including tears, fading, scratches, and discoloration and colorize the photo, adds realistic colors",
    },
    "enhance_photo": {
        "display_name": "Enhance photo",
        "prompt": "Enhance image quality while preserving details.",
    },
    "none": {"display_name": "None", "prompt": ""},
}


def get_edit_prompts():
    try:
        with open(Path(get_configs_path()) / "edit_image_prompts.json") as file:
            prompts = json.load(file)
    except Exception as e:
        print(f"Error loading edit prompts: {e} \nloading default edit prompts")
        return default_edit_prompts
    return prompts
