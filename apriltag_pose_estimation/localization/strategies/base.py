import abc
from typing import Sequence, Optional, Any, TYPE_CHECKING

from apriltag_pose_estimation.core import AprilTagDetection, CameraParameters, Transform

if TYPE_CHECKING:
    from apriltag_pose_estimation.localization import AprilTagField
else:
    AprilTagField = Any


__all__ = [
    'CameraLocalizationStrategy'
]


class CameraLocalizationStrategy(abc.ABC):
    """
    Abstract base class (ABC) for all localization strategies.

    A strategy has a single method ``estimate_robot_pose`` which estimates the pose of the world origin in the camera
    frame based on detected AprilTag(s).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def estimate_world_to_camera(self,
                                 detections: Sequence[AprilTagDetection],
                                 field: AprilTagField,
                                 camera_params: CameraParameters) -> Optional[Transform]:
        """
        Estimates a Euclidean transformation from the world frame to the camera frame based on the detected AprilTag(s).

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

    def __str__(self) -> str:
        return f'<{type(self).__qualname__} object \'{self.name}\' at {id(self):#x}>'

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'
