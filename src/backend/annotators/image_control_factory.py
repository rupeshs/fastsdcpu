from backend.annotators.canny_control import CannyControl
from backend.annotators.pose_control import PoseControl


class ImageControlFactory:
    def create_control(self, controlnet_type: str):
        if controlnet_type == "Canny":
            return CannyControl()
        elif controlnet_type == "Pose":
            return PoseControl()
        else:
            print("Error: Control type not implemented!")
            raise Exception("Error: Control type not implemented!")
