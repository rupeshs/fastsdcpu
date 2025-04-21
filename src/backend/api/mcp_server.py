import platform

import uvicorn
from backend.device import get_device_name
from backend.models.device import DeviceInfo
from constants import APP_VERSION, DEVICE
from context import Context
from fastapi import FastAPI, Request
from fastapi_mcp import FastApiMCP
from state import get_settings
from fastapi.middleware.cors import CORSMiddleware
from models.interface_types import InterfaceType
from fastapi.staticfiles import StaticFiles

app_settings = get_settings()
app = FastAPI(
    title="FastSD CPU",
    description="Fast stable diffusion on CPU",
    version=APP_VERSION,
    license_info={
        "name": "MIT",
        "identifier": "MIT",
    },
    describe_all_responses=True,
    describe_full_response_schema=True,
)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

context = Context(InterfaceType.API_SERVER)
app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get(
    "/info",
    description="Get system information",
    summary="Get system information",
    operation_id="get_system_info",
)
async def info() -> dict:
    device_info = DeviceInfo(
        device_type=DEVICE,
        device_name=get_device_name(),
        os=platform.system(),
        platform=platform.platform(),
        processor=platform.processor(),
    )
    return device_info.model_dump()


@app.post(
    "/generate",
    description="Generate image from text prompt",
    summary="Text to image generation",
    operation_id="generate",
)
async def generate(
    prompt: str,
    request: Request,
) -> str:
    """
    Returns URL of the generated image for text prompt
    """
    app_settings.settings.lcm_diffusion_setting.prompt = prompt
    images = context.generate_text_to_image(app_settings.settings)
    image_names = context.save_images(
        images,
        app_settings.settings,
    )
    url = request.url_for("results", path=image_names[0])
    image_url = f"The generated image available at the URL {url}"
    return image_url


def start_mcp_server(port: int = 8000):
    print(f"Starting MCP server on port {port}...")
    mcp = FastApiMCP(
        app,
        name="FastSDCPU MCP",
        description="MCP server for FastSD CPU API",
    )

    mcp.mount()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
