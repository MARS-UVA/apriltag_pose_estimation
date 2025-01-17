from collections.abc import Sequence
from typing import Optional, List

from ..estimation import PoseEstimationStrategy
from ...core.camera import CameraParameters
from ...core.detection import AprilTagDetection
from ...core.euclidean import Pose
from ...core.field import AprilTagField
from ...core.pnp import PnPMethod, solve_pnp


__all__ = ['LowestAmbiguityEstimationStrategy']


class LowestAmbiguityEstimationStrategy(PoseEstimationStrategy):
    """
    An estimation strategy which solves the Perspective-N-Point problem for each AprilTag's corner points and chooses
    the estimate with the lowest ambiguity.

    This strategy is implemented with OpenCV's solvePnP function. See
    https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html for more information.

    Each of the corners of the detected AprilTags is computed in the world frame, and these points are passed for each
    detected AprilTag individually into the PnP solver.

    This implementation derives heavily from MultiTag pose estimation in PhotonVision (see
    https://github.com/PhotonVision/photonvision/blob/main/photon-targeting/src/main/java/org/photonvision/estimation/OpenCVHelp.java#L465).
    """
    def __init__(self, pnp_method: PnPMethod = PnPMethod.SQPNP):
        """
        :param pnp_method: A method the strategy will use to solve the Perspective-N-Point problem. Cannot be
                           ``PnPMethod.IPPE``. Defaults to ``PnPMethod.SQPNP``.
        """
        super().__init__()
        self.__pnp_method = pnp_method

    def estimate_pose(self, detections: Sequence[AprilTagDetection], field: AprilTagField,
                      camera_params: CameraParameters) -> Optional[Pose]:
        if not detections:
            return None
        pose_candidates: List[Pose] = []
        for detection in detections:
            object_points = field.get_corners(detection.tag_id)
            image_points = detection.corners
            poses = solve_pnp(object_points, image_points, camera_params, method=self.__pnp_method)
            if len(poses) == 1:
                best_pose = poses[0]
            else:
                best_pose = poses[0].with_ambiguity(poses[0].error / poses[1].error)
            pose_candidates.append(best_pose)
        if not pose_candidates:
            return

        return min(pose_candidates, key=lambda pose: pose.ambiguity if pose.ambiguity is not None else float('inf'))
