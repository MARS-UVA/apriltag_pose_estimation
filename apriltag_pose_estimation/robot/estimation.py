import abc
from collections.abc import Sequence
from typing import Optional, Dict

from ..core.camera import CameraParameters
from ..core.detection import AprilTagDetection
from ..core.euclidean import Pose
from ..core.field import AprilTagField


__all__ = ['RobotPoseEstimationStrategy']


class RobotPoseEstimationStrategy(abc.ABC):
    """
    Abstract base class (ABC) for all robot pose estimation strategies.

    A strategy has a single method ``estimate_robot_pose`` which estimates the pose of the robot in the world frame
    based on detected AprilTag(s).
    """
    @abc.abstractmethod
    def estimate_robot_pose(self,
                            detections: Sequence[AprilTagDetection],
                            field: AprilTagField,
                            camera_params: CameraParameters,
                            camera_pose_on_robot: Pose) -> Optional[Pose]:
        """
        Estimates the pose of the robot in the world frame based on the detected AprilTag(s).
        :param detections: A sequence of ``AprilTagDetection`` objects created from images by the camera described by
                           ``camera_params`` and ``camera_pose_on_robot``.
        :param field: An :class:`AprilTagField` describing the positions of all the AprilTags on the field in the
                      world frame.
        :param camera_params: Characteristic parameters for the camera being used to detect AprilTags.
        :param camera_pose_on_robot:
        :return:
        """
        pass
