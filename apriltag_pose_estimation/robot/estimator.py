from typing import Optional

import numpy as np
import numpy.typing as npt
from pupil_apriltags import Detector

from .estimation import RobotPoseEstimationStrategy
from apriltag_pose_estimation.core.field import Field
from apriltag_pose_estimation.core.detection import AprilTagDetection
from apriltag_pose_estimation.core.euclidean import Pose
from apriltag_pose_estimation.core.camera import CameraParameters


class RobotPoseEstimator:
    def __init__(self, strategy: RobotPoseEstimationStrategy, field: Field, camera_params: CameraParameters, camera_pose_on_robot: Pose, **detector_kwargs):
        self.__strategy = strategy
        self.__field = field
        self.__camera_params = camera_params
        self.__camera_pose_on_robot = camera_pose_on_robot
        self.__detector = Detector(families=self.__field.tag_family, **detector_kwargs)

    def estimate_robot_pose(self, image: npt.NDArray[np.uint8]) -> Optional[Pose]:
        detections = [AprilTagDetection(tag_id=detection.tag_id,
                                        tag_family=detection.tag_family.decode('utf-8'),
                                        center=detection.center,
                                        corners=detection.corners,
                                        decision_margin=detection.decision_margin,
                                        hamming=detection.hamming)
                      for detection in self.__detector.detect(img=image, estimate_tag_pose=False)  # type: ignore
                      if detection.tag_id in self.__field and detection.tag_family == self.__field.tag_family]

        if not detections:
            return

        return self.__strategy.estimate_robot_pose(detections,
                                                   field=self.__field,
                                                   camera_params=self.__camera_params,
                                                   camera_pose_on_robot=self.__camera_pose_on_robot)
