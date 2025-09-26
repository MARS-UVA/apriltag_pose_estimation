"""Defines a class for localizing a camera with a field of AprilTags."""

from dataclasses import dataclass
from typing import Optional, List, Unpack

import numpy as np
import numpy.typing as npt

from .strategies.base import CameraLocalizationStrategy
from .field import AprilTagField
from ..core.camera import CameraParameters
from ..core.detection import AprilTagDetection, AprilTagDetector, AprilTagDetectorParams
from ..core.euclidean import Transform


__all__ = ['CameraLocalizer']


class CameraLocalizer:
    """
    A localizer for a camera based on positions of known AprilTags in images taken by the camera.

    Localizers take information about the field and camera as arguments, along with arguments for the AprilTag detector.
    They also need a :class:`CameraLocalizationStrategy`, which tells the localizer how to determine the pose of the
    camera from a sequence of detected AprilTags.
    """

    @dataclass(frozen=True)
    class Result:
        """Result of an AprilTag-based camera localization."""

        estimated_pose: Optional[Transform]
        """
        The estimated pose of the camera. Will be ``None`` if no AprilTags were detected or the camera localization
        strategy refused to make an estimate.
        """
        detections: List[AprilTagDetection]
        """List of all detected AprilTags and information about them."""

    def __init__(self,
                 strategy: CameraLocalizationStrategy,
                 field: AprilTagField,
                 camera_params: CameraParameters,
                 **detector_kwargs: Unpack[AprilTagDetectorParams]):
        """
        :param strategy: The strategy for this estimator to use.
        :param field: An :py:class:`~apriltag_pose_estimation.localization.field.AprilTagField` describing the positions
           of all the AprilTags on the field in the world frame.
        :param camera_params: Characteristic parameters for the camera being used to detect AprilTags.
        :param detector_kwargs: Arguments to pass to the AprilTag detector (see :class:`AprilTagDetector`).
        """
        self.__strategy = strategy
        self.__field = field
        self.__camera_params = camera_params
        self.__detector = AprilTagDetector(families=self.__field.tag_family, **detector_kwargs)

    @property
    def strategy(self) -> CameraLocalizationStrategy:
        """Strategy this estimator will use."""
        return self.__strategy

    @property
    def field(self) -> AprilTagField:
        """
        An :py:class:`~apriltag_pose_estimation.localization.field.AprilTagField` describing the positions of all the
        AprilTags on the field in the world frame.
        """
        return self.__field

    @property
    def camera_params(self) -> CameraParameters:
        """Parameters for the camera from which images will be provided."""
        return self.__camera_params

    def estimate_pose(self, image: npt.NDArray[np.uint8]) -> 'CameraLocalizer.Result':
        """
        Estimates a Euclidean transformation from the world frame to the camera frame based on AprilTag(s) detected in
        the given image.

        This function returns the estimated pose of the *world origin* in the *camera frame*, not the other way around.
        While this may seem unintuitive, it ends up being more useful when transforming the pose later.

        :param image: An image containing AprilTags taken by the camera described by this estimator's camera parameters
                      and camera pose.
        :return: The estimated pose of the world origin in the camera frame, or ``None`` if an estimate could not be
                 made.
        """
        detections = [detection
                      for detection in self.__detector.detect(img=image,
                                                              camera_params=self.__camera_params,
                                                              tag_size=self.__field.tag_size,
                                                              estimate_tag_pose=True)
                      if detection.tag_id in self.__field and detection.tag_family == self.__field.tag_family]

        return self.Result(estimated_pose=self.__strategy.estimate_world_to_camera(detections,
                                                                                   field=self.__field,
                                                                                   camera_params=self.__camera_params),
                           detections=detections)
