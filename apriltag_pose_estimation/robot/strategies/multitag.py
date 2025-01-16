from operator import attrgetter
from typing import Iterable, List, Dict, Optional

import numpy as np
import numpy.typing as npt

from ...core.camera import CameraParameters
from ...core.euclidean import Pose
from ...core.detection import AprilTagDetection
from ..estimation import RobotPoseEstimationStrategy
from ...core.field import Field
from ...core.pnp import PnPMethod, solve_pnp


class MultiTagEstimationStrategy(RobotPoseEstimationStrategy):
    def __init__(self, fallback_strategy: Optional[RobotPoseEstimationStrategy] = None):
        super().__init__()
        if isinstance(fallback_strategy, MultiTagEstimationStrategy):
            raise TypeError('multitag fallback strategy cannot be multitag')
        self.__corners_cache: Dict[Field, Dict[int, npt.NDArray[np.float64]]] = {}
        self.__fallback_strategy = fallback_strategy
        
    def set_field(self, field: Field) -> None:
        super().set_field(field)
        if self.__fallback_strategy is not None:
            self.__fallback_strategy.set_field(field)

    def estimate_robot_pose(self, detections: Iterable[AprilTagDetection], field: Field,
                            camera_params: CameraParameters, camera_pose_on_robot: Pose) -> Optional[Pose]:
        object_corner_groups: List[npt.NDArray[np.float64]] = []
        for detection in detections:
            if detection.tag_id in self.__corners_cache:
                object_corner_groups.append(self.__corners_cache[detection.tag_id])
            else:
                corners = _get_corners_from_pose(field[detection.tag_id], field.tag_size)
                self.__corners_cache[detection.tag_id] = corners
                object_corner_groups.append(corners)

        object_points = np.vstack(object_corner_groups)
        image_points = np.vstack([detection.corners for detection in detections])

        poses = solve_pnp(object_points, image_points, camera_params, method=PnPMethod.SQPNP)
        if not poses:
            return

        origin_pose_in_camera = min(poses, key=attrgetter('error'))

        return Pose.from_matrix(np.linalg.inv(camera_pose_on_robot.get_matrix() @ origin_pose_in_camera.get_matrix()),
                                error=origin_pose_in_camera.error)

    def __use_fallback_strategy(self, detections: Iterable[AprilTagDetection], field: Field,
                                camera_params: CameraParameters, camera_pose_on_robot: Pose) -> Optional[Pose]:
        if self.__fallback_strategy is None:
            return None
        return self.estimate_robot_pose(detections, field, camera_params, camera_pose_on_robot)


def _get_corners_from_pose(pose: Pose, tag_size: float) -> npt.NDArray[np.float64]:
    corner_points = np.array([
            [-1, +1, 0, 1],
            [+1, +1, 0, 1],
            [+1, -1, 0, 1],
            [-1, -1, 0, 1],
        ]) / 2 * tag_size
    return pose.get_matrix() @ corner_points
