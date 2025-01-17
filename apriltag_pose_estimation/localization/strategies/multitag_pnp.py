from collections.abc import Sequence
from operator import attrgetter
from typing import Optional

import numpy as np

from ..estimation import PoseEstimationStrategy
from ...core.camera import CameraParameters
from ...core.detection import AprilTagDetection
from ...core.euclidean import Pose
from ...core.field import AprilTagField
from ...core.pnp import PnPMethod, solve_pnp


__all__ = ['MultiTagPnPEstimationStrategy']


class MultiTagPnPEstimationStrategy(PoseEstimationStrategy):
    """
    An estimation strategy which solves the Perspective-N-Point problem.

    This strategy is implemented with OpenCV's solvePnP function. See
    https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html for more information.

    Each of the corners of the detected AprilTags is computed in the world frame, and these points are passed into the
    PnP solver. If there is only one detected AprilTag or if the PnP solver fails, a fallback strategy was used.
    """
    def __init__(self, fallback_strategy: Optional[PoseEstimationStrategy] = None):
        """
        :param fallback_strategy: A strategy to use if only one AprilTag was detected or the PnP solver fails. Cannot
                                  be a :class:`MultiTagPnPEstimationStrategy`.
        """
        super().__init__()
        if isinstance(fallback_strategy, MultiTagPnPEstimationStrategy):
            raise TypeError('multitag fallback strategy cannot be multitag')
        self.__fallback_strategy = fallback_strategy

    def estimate_pose(self, detections: Sequence[AprilTagDetection], field: AprilTagField,
                      camera_params: CameraParameters) -> Optional[Pose]:
        if not detections:
            return None
        if len(detections) == 1:
            return self.__use_fallback_strategy(detections, field, camera_params)

        object_points = field.get_corners(*(detection.tag_id for detection in detections))
        image_points = np.vstack([detection.corners for detection in detections])

        poses = solve_pnp(object_points, image_points, camera_params, method=PnPMethod.SQPNP)
        if not poses:
            return self.__use_fallback_strategy(detections, field, camera_params)

        return min(poses, key=attrgetter('error'))

    def __use_fallback_strategy(self, detections: Sequence[AprilTagDetection], field: AprilTagField,
                                camera_params: CameraParameters) -> Optional[Pose]:
        if self.__fallback_strategy is None:
            return None
        return self.estimate_pose(detections, field, camera_params)
