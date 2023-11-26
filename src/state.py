from app_settings import AppSettings
from typing import Optional


class _AppState:
    _instance: Optional["_AppState"] = None
    settings: Optional[AppSettings] = None


def get_state() -> _AppState:
    if _AppState._instance is None:
        _AppState._instance = _AppState()
    return _AppState._instance


def get_settings(skip_file: bool = False) -> AppSettings:
    state = get_state()
    if state.settings is None:
        state.settings = AppSettings()
        state.settings.load(skip_file)
    return state.settings
