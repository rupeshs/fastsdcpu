from app_settings import AppSettings
from typing import Optional

from context import Context
from models.interface_types import InterfaceType
from backend.safety_checker import SafetyChecker


class _AppState:
    _instance: Optional["_AppState"] = None
    settings: Optional[AppSettings] = None
    context: Optional[Context] = None
    safety_checker: Optional[SafetyChecker] = None


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


def get_context(interface_type: InterfaceType) -> Context:
    state = get_state()
    if state.context is None:
        state.context = Context(interface_type)
    return state.context


def get_safety_checker() -> SafetyChecker:
    state = get_state()
    if state.safety_checker is None:
        print("Initializing safety checker")
        state.safety_checker = SafetyChecker()
    return state.safety_checker
