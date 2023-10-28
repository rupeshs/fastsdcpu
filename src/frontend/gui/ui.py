from typing import List
from frontend.gui.app_window import MainWindow
from PyQt5.QtWidgets import QApplication
import sys
from app_settings import AppSettings


def start_gui(
    argv: List[str],
    app_settings: AppSettings,
):
    app = QApplication(sys.argv)
    window = MainWindow(app_settings)
    window.show()
    app.exec()
