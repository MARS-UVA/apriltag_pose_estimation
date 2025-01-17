from typing import Optional

import numpy as np
import numpy.typing as npt
from pupil_apriltags import Detector

from .estimation import RobotPoseEstimationStrategy
from ..core.camera import CameraParameters
from ..core.detection import AprilTagDetection
from ..core.euclidean import Pose
from ..core.field import AprilTagField


__all__ = ['RobotPoseEstimator']


class RobotPoseEstimator:
    """
    An estimator for the position of a robot equipped with a camera detecting AprilTags.

    Estimators take information about the field and camera as arguments, along with arguments for the AprilTag detector.
    They also need a :class:`RobotPoseEstimationStrategy`, which tells the estimator how to determine the position of
    the robot from a sequence of detected AprilTags.
    """
    def __init__(self,
                 strategy: RobotPoseEstimationStrategy,
                 field: AprilTagField,
                 camera_params: CameraParameters,
                 camera_pose_on_robot: Pose,
                 **detector_kwargs):
        """
        :param strategy: The strategy for this estimator to use.
        :param field: An :class:`AprilTagField` describing the positions of all the AprilTags on the field in the
                      world frame.
        :param camera_params: Characteristic parameters for the camera being used to detect AprilTags.
        :param camera_pose_on_robot: The pose of the camera relative to the robot's base_link frame (see
                                     https://www.ros.org/reps/rep-0105.html#coordinate-frames for more info).
        :param detector_kwargs: Arguments to pass to the AprilTag detector (see :class:`AprilTagDetector`).
        """
        self.__strategy = strategy
        self.__field = field
        self.__camera_params = camera_params
        self.__camera_pose_on_robot = camera_pose_on_robot
        self.__detector = Detector(families=self.__field.tag_family, **detector_kwargs)

    def estimate_robot_pose(self, image: npt.NDArray[np.uint8]) -> Optional[Pose]:
        """
        Estimates the pose of the robot in the world frame based on AprilTags in the given image.
        :param image: An image containing AprilTags taken by the camera described by this estimator's camera parameters
                      and camera pose.
        :return: The estimated pose of the robot in the world frame, or ``None`` if an estimate could not be made.
        """
        detections = [AprilTagDetection(tag_id=detection.tag_id,
                                        tag_family=detection.tag_family.decode('utf-8'),
                                        center=detection.center,
                                        corners=detection.corners,
                                        decision_margin=detection.decision_margin,
                                        hamming=detection.hamming)
                      for detection in self.__detector.detect(img=image, estimate_tag_pose=False)  # type: ignore
                      if detection.tag_id in self.__field and detection.tag_family == self.__field.tag_family]

        return self.__strategy.estimate_robot_pose(detections,
                                                   field=self.__field,
                                                   camera_params=self.__camera_params,
                                                   camera_pose_on_robot=self.__camera_pose_on_robot)
