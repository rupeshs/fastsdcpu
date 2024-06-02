import platform

import uvicorn
from backend.api.models.response import StableDiffusionResponse
from backend.models.device import DeviceInfo
from backend.base64_image import base64_image_to_pil, pil_image_to_base64_str
from backend.device import get_device_name
from backend.models.lcmdiffusion_setting import DiffusionTask, LCMDiffusionSetting
from constants import APP_VERSION, DEVICE
from context import Context
from fastapi import FastAPI
from models.interface_types import InterfaceType
from state import get_settings

app_settings = get_settings()
app = FastAPI(
    title="FastSD CPU",
    description="Fast stable diffusion on CPU",
    version=APP_VERSION,
    license_info={
        "name": "MIT",
        "identifier": "MIT",
    },
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
print(app_settings.settings.lcm_diffusion_setting)

context = Context(InterfaceType.API_SERVER)


@app.get("/api/")
async def root():
    return {"message": "Welcome to FastSD CPU API"}


@app.get(
    "/api/info",
    description="Get system information",
    summary="Get system information",
)
async def info():
    device_info = DeviceInfo(
        device_type=DEVICE,
        device_name=get_device_name(),
        os=platform.system(),
        platform=platform.platform(),
        processor=platform.processor(),
    )
    return device_info.model_dump()


@app.get(
    "/api/config",
    description="Get current configuration",
    summary="Get configurations",
)
async def config():
    return app_settings.settings


@app.get(
    "/api/models",
    description="Get available models",
    summary="Get available models",
)
async def models():
    return {
        "lcm_lora_models": app_settings.lcm_lora_models,
        "stable_diffusion": app_settings.stable_diffsuion_models,
        "openvino_models": app_settings.openvino_lcm_models,
        "lcm_models": app_settings.lcm_models,
    }


@app.post(
    "/api/generate",
    description="Generate image(Text to image,Image to Image)",
    summary="Generate image(Text to image,Image to Image)",
)
async def generate(diffusion_config: LCMDiffusionSetting) -> StableDiffusionResponse:
    app_settings.settings.lcm_diffusion_setting = diffusion_config
    if diffusion_config.diffusion_task == DiffusionTask.image_to_image:
        app_settings.settings.lcm_diffusion_setting.init_image = base64_image_to_pil(
            diffusion_config.init_image
        )

    images = context.generate_text_to_image(app_settings.settings)

    images_base64 = [pil_image_to_base64_str(img) for img in images]
    return StableDiffusionResponse(
        latency=round(context.latency, 2),
        images=images_base64,
    )


def start_web_server():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
