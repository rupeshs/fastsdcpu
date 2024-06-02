from enum import Enum


class InterfaceType(Enum):
    WEBUI = "Web User Interface"
    GUI = "Graphical User Interface"
    CLI = "Command Line Interface"
    API_SERVER = "API Server"
