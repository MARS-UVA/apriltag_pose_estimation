from operator import attrgetter
from typing import List

import numpy as np
import numpy.typing as npt
from pupil_apriltags import Detector, Detection

from ..estimation import AprilTagPoseEstimationStrategy
from ...core.detection import AprilTagDetection
from ...core.camera import CameraParameters
from ...core.euclidean import Pose
from ...core.pnp import PnPMethod, solve_pnp


__all__ = ['PerspectiveNPointStrategy']


class PerspectiveNPointStrategy(AprilTagPoseEstimationStrategy):
    """
    A pose estimation strategy which uses OpenCV's solvePnP function.

    See also: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
    """
    def __init__(self, method: PnPMethod = PnPMethod.ITERATIVE):
        self.__method = method

    @property
    def method(self):
        """The method which this strategy is using for Perspective-N-Point."""
        return self.__method

    def estimate_tag_pose(self,
                          image: npt.NDArray[np.uint8],
                          detector: Detector,
                          camera_params: CameraParameters,
                          tag_size: float) -> List[AprilTagDetection]:
        detection: Detection
        # noinspection PyTypeChecker
        return [AprilTagDetection(tag_id=detection.tag_id,
                                  tag_family=detection.tag_family,
                                  center=detection.center,
                                  corners=detection.corners,
                                  decision_margin=detection.decision_margin,
                                  hamming=detection.hamming,
                                  tag_poses=self.__get_poses_from_corners(detection, camera_params, tag_size))
                for detection in detector.detect(image)]

    @property
    def name(self):
        return f'pnp-{self.__method.name.lower()}'

    def __get_poses_from_corners(self,
                                 detection: Detection,
                                 camera_params: CameraParameters,
                                 tag_size: float) -> List[Pose]:
        object_points = np.array([
            [-1, +1, 0],
            [+1, +1, 0],
            [+1, -1, 0],
            [-1, -1, 0],
        ]) / 2 * tag_size
        image_points = detection.corners

        return sorted(solve_pnp(object_points, image_points, camera_params, method=self.__method),
                      key=attrgetter('error'))
