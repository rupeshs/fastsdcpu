from pydantic import BaseModel


class DeviceInfo(BaseModel):
    device_type: str
    device_name: str
    os: str
    platform: str
    processor: str
