from frontend.gui.ui import start_gui
from app_settings import AppSettings
import platform

try:
    print(f"Running on {platform.system()} platform")
    print(f"OS: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
except Exception as ex:
    print(f"Error ocurred while getting system information {ex}")

app_settings = AppSettings()
app_settings.load()
start_gui([], app_settings)
