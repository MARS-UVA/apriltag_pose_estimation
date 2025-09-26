"""Defines the base class for all pose estimation strategies."""

import abc
from typing import List

import numpy as np
from numpy import typing as npt

from apriltag_pose_estimation.core import AprilTagDetector, CameraParameters, AprilTagDetection


__all__ = [
    'AprilTagPoseEstimationStrategy'
]


class AprilTagPoseEstimationStrategy(abc.ABC):
    """Abstract base class for strategies to estimate the positions of AprilTags relative to a camera."""
    @abc.abstractmethod
    def estimate_tag_pose(self,
                          image: npt.NDArray[np.uint8],
                          detector: AprilTagDetector,
                          camera_params: CameraParameters,
                          tag_size: float) -> List[AprilTagDetection]:
        """
        Estimates the poses of all detectable AprilTags in the provided image.
        :param image: The image in which tags will be detected.
        :param detector: An AprilTagDetector instance.
        :param camera_params: Parameters of the camera used to take the image.
        :param tag_size: The size of the tag, in meters.
        :return: A list of all AprilTags which were detected. If there were no AprilTags detected, an empty list is
                 returned.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self):
        """A name for this strategy."""
        pass
