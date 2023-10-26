from frontend.gui.ui import start_gui
from app_settings import AppSettings
from utils import show_system_info
from frontend.webui.ui import start_webui


show_system_info()
app_settings = AppSettings()
app_settings.load()
start_gui([], app_settings)
# start_webui()
