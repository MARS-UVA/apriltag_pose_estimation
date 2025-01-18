from collections.abc import Sequence
from typing import Optional

import numpy as np

from ..estimation import PoseEstimationStrategy
from ...core.camera import CameraParameters
from ...core.detection import AprilTagDetection
from ...core.euclidean import Pose
from ...core.exceptions import EstimationError
from ...core.field import AprilTagField
from ...core.pnp import PnPMethod, solve_pnp


__all__ = ['MultiTagPnPEstimationStrategy']


class MultiTagPnPEstimationStrategy(PoseEstimationStrategy):
    """
    An estimation strategy which solves the Perspective-N-Point problem across all detected AprilTag corner points.

    This strategy is implemented with OpenCV's solvePnP function. See
    https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html for more information.

    Each of the corners of the detected AprilTags is computed in the world frame, and these points are passed into the
    PnP solver. If there is only one detected AprilTag or if the PnP solver fails, a fallback strategy is used.

    This implementation derives heavily from MultiTag pose estimation in PhotonVision (see
    https://github.com/PhotonVision/photonvision/blob/main/photon-targeting/src/main/java/org/photonvision/estimation/OpenCVHelp.java#L465).
    """
    def __init__(self, fallback_strategy: Optional[PoseEstimationStrategy] = None,
                 pnp_method: PnPMethod = PnPMethod.SQPNP):
        """
        :param fallback_strategy: A strategy to use if only one AprilTag was detected or the PnP solver fails. Cannot
                                  be a :class:`MultiTagPnPEstimationStrategy`.
        :param pnp_method: A method the strategy will use to solve the Perspective-N-Point problem. Cannot be
                           ``PnPMethod.IPPE``. Defaults to ``PnPMethod.SQPNP``.
        """
        super().__init__()
        if isinstance(fallback_strategy, MultiTagPnPEstimationStrategy):
            raise TypeError('multitag fallback strategy cannot be another multitag PnP strategy')
        if pnp_method is PnPMethod.IPPE:
            raise ValueError('PnP method cannot be IPPE for multitag estimation')
        self.__fallback_strategy = fallback_strategy
        self.__pnp_method = pnp_method

    def estimate_pose(self, detections: Sequence[AprilTagDetection], field: AprilTagField,
                      camera_params: CameraParameters) -> Optional[Pose]:
        if not detections:
            return None
        if len(detections) == 1:
            return self.__use_fallback_strategy(detections, field, camera_params)

        object_points = field.get_corners(*(detection.tag_id for detection in detections))
        image_points = np.vstack([detection.corners for detection in detections])

        try:
            poses = solve_pnp(object_points, image_points, camera_params, method=self.__pnp_method)
        except EstimationError:
            return self.__use_fallback_strategy(detections, field, camera_params)
        if not poses:
            return self.__use_fallback_strategy(detections, field, camera_params)

        return poses[0]

    def __use_fallback_strategy(self, detections: Sequence[AprilTagDetection], field: AprilTagField,
                                camera_params: CameraParameters) -> Optional[Pose]:
        if self.__fallback_strategy is None:
            return None
        return self.estimate_pose(detections, field, camera_params)
