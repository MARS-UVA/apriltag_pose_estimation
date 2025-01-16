import abc
from collections.abc import Sequence
from typing import Optional, Dict

from ..core.camera import CameraParameters
from ..core.detection import AprilTagDetection
from ..core.euclidean import Pose
from ..core.field import AprilTagField


__all__ = ['RobotPoseEstimationStrategy']


class RobotPoseEstimationStrategy(abc.ABC):
    @abc.abstractmethod
    def estimate_robot_pose(self,
                            detections: Sequence[AprilTagDetection],
                            field: AprilTagField,
                            camera_params: CameraParameters,
                            camera_pose_on_robot: Pose) -> Optional[Pose]:
        pass
