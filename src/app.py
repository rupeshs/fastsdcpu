from app_settings import AppSettings
from utils import show_system_info
import constants
from argparse import ArgumentParser

parser = ArgumentParser(description=f"FAST SD CPU {constants.APP_VERSION}")
parser.add_argument(
    "-s",
    "--share",
    action="store_true",
    help="Create sharable link(Web UI)",
    required=False,
)
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument(
    "-g",
    "--gui",
    action="store_true",
    help="Start desktop GUI",
)
group.add_argument(
    "-w",
    "--webui",
    action="store_true",
    help="Start Web UI",
)
args = parser.parse_args()

parser.print_help()
show_system_info()
print(f"Using device : {constants.DEVICE}")

app_settings = AppSettings()
app_settings.load()

if args.gui:
    from frontend.gui.ui import start_gui

    print("Starting in desktop GUI mode(Qt)")
    start_gui(
        [],
        app_settings,
    )
elif args.webui:
    from frontend.webui.ui import start_webui

    print("Starting in Web UI mode")
    start_webui(
        app_settings,
        args.share,
    )
