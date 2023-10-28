from frontend.gui.ui import start_gui
from app_settings import AppSettings

app_settings = AppSettings()
app_settings.load()

start_gui([], app_settings)
