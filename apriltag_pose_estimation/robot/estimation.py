import abc
from collections.abc import Iterable
from typing import Optional, Dict

from ..core.field import Field
from ..core.camera import CameraParameters
from ..core.euclidean import Pose
from ..core.detection import AprilTagDetection


class RobotPoseEstimationStrategy(abc.ABC):
    def __init__(self):
        self.__corners: Optional[Dict[int, Pose]] = None

    def set_field(self, field: Field) -> None:
        pass


    @abc.abstractmethod
    def estimate_robot_pose(self,
                            detections: Iterable[AprilTagDetection],
                            field: Field,
                            camera_params: CameraParameters,
                            camera_pose_on_robot: Pose) -> Optional[Pose]:
        pass
