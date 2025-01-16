from collections.abc import Sequence
from operator import attrgetter
from typing import Iterable, List, Dict, Optional

import numpy as np
import numpy.typing as npt

from ..estimation import RobotPoseEstimationStrategy
from ...core.camera import CameraParameters
from ...core.detection import AprilTagDetection
from ...core.euclidean import Pose
from ...core.field import AprilTagField
from ...core.pnp import PnPMethod, solve_pnp


__all__ = ['MultiTagPnPEstimationStrategy']


class MultiTagPnPEstimationStrategy(RobotPoseEstimationStrategy):
    def __init__(self, fallback_strategy: Optional[RobotPoseEstimationStrategy] = None):
        super().__init__()
        if isinstance(fallback_strategy, MultiTagPnPEstimationStrategy):
            raise TypeError('multitag fallback strategy cannot be multitag')
        self.__fallback_strategy = fallback_strategy

    def estimate_robot_pose(self, detections: Sequence[AprilTagDetection], field: AprilTagField,
                            camera_params: CameraParameters, camera_pose_on_robot: Pose) -> Optional[Pose]:
        if not detections:
            return None
        if len(detections) == 1:
            return self.__use_fallback_strategy(detections, field, camera_params, camera_pose_on_robot)

        object_points = field.get_corners(*(detection.tag_id for detection in detections))
        image_points = np.vstack([detection.corners for detection in detections])

        poses = solve_pnp(object_points, image_points, camera_params, method=PnPMethod.SQPNP)
        if not poses:
            return self.__use_fallback_strategy(detections, field, camera_params, camera_pose_on_robot)

        origin_pose_in_camera = min(poses, key=attrgetter('error'))

        return Pose.from_matrix(np.linalg.inv(camera_pose_on_robot.get_matrix() @ origin_pose_in_camera.get_matrix()),
                                error=origin_pose_in_camera.error)

    def __use_fallback_strategy(self, detections: Sequence[AprilTagDetection], field: AprilTagField,
                                camera_params: CameraParameters, camera_pose_on_robot: Pose) -> Optional[Pose]:
        if self.__fallback_strategy is None:
            return None
        return self.estimate_robot_pose(detections, field, camera_params, camera_pose_on_robot)
