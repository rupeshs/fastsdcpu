from abc import ABC, abstractmethod

from PIL import Image


class ControlInterface(ABC):
    @abstractmethod
    def get_control_image(
        self,
        image: Image,
    ) -> Image:
        pass
