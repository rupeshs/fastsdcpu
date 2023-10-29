from os import path, mkdir
from typing import Any
from uuid import uuid4
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
import json


class ImageSaver:
    @staticmethod
    def save_images(
        output_path: str,
        images: Any,
        folder_name: str = "",
        format: str = ".png",
        lcm_diffusion_setting: LCMDiffusionSetting = None,
    ) -> None:
        gen_id = uuid4()
        for index, image in enumerate(images):
            if not path.exists(output_path):
                mkdir(output_path)

            if folder_name:
                out_path = path.join(
                    output_path,
                    folder_name,
                )
            else:
                out_path = output_path

            if not path.exists(out_path):
                mkdir(out_path)
            image.save(path.join(out_path, f"{gen_id}-{index+1}{format}"))
        if lcm_diffusion_setting:
            with open(path.join(out_path, f"{gen_id}.json"), "w") as json_file:
                json.dump(
                    lcm_diffusion_setting.model_dump(),
                    json_file,
                    indent=4,
                )
