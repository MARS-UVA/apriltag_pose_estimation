import abc
from collections.abc import Sequence
from typing import Optional, Dict

from ..core.camera import CameraParameters
from ..core.detection import AprilTagDetection
from ..core.euclidean import Pose
from ..core.field import AprilTagField


__all__ = ['PoseEstimationStrategy']


class PoseEstimationStrategy(abc.ABC):
    """
    Abstract base class (ABC) for all robot pose estimation strategies.

    A strategy has a single method ``estimate_robot_pose`` which estimates the pose of the world origin in the camera
    frame based on detected AprilTag(s).
    """
    @abc.abstractmethod
    def estimate_pose(self,
                      detections: Sequence[AprilTagDetection],
                      field: AprilTagField,
                      camera_params: CameraParameters) -> Optional[Pose]:
        """
        Estimates the pose of the camera in the world frame based on the detected AprilTag(s).

        This function returns the estimated pose of the *world origin* in the *camera frame*, not the other way around.
        While this may seem unintuitive, it ends up being more useful when transforming the pose later.

        Implementors must be careful to return the world origin pose in the camera frame.

        :param detections: A sequence of ``AprilTagDetection`` objects created from images by the camera described by
                           ``camera_params``.
        :param field: An :class:`AprilTagField` describing the positions of all the AprilTags on the field in the
                      world frame.
        :param camera_params: Characteristic parameters for the camera being used to detect AprilTags.
        :return: The estimated pose of the world origin in the camera frame, or ``None`` if an estimate could not be
                 made.
        """
        pass
