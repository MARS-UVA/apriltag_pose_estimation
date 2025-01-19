from collections.abc import Sequence, Callable
from typing import Optional, List

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..estimation import PoseEstimationStrategy
from ...core import EstimationError
from ...core.camera import CameraParameters
from ...core.detection import AprilTagDetection
from ...core.euclidean import Pose
from ...core.field import AprilTagField
from ...core.pnp import PnPMethod, solve_pnp


__all__ = ['MultiTagSpecialEstimationStrategy']


class MultiTagSpecialEstimationStrategy(PoseEstimationStrategy):
    """
    An estimation strategy which attempts to resolve ambiguous tag poses to compile a more accurate result.

    Unlike other strategies, this strategy requires knowledge of the camera's angle in the world frame (passed as the
    world origin's rotation in the camera frame). Each of the corners of the detected AprilTags is computed in the world
    frame, and each set of corners are used to solve the PnP problem to determine candidate poses for the world origin.
    Then only the solutions for each AprilTag which are closest to the actual angle kept, and their translations and
    rotations are averaged to produce a final estimate.

    If there is only one detected AprilTag or if the PnP solver fails on all AprilTags, a fallback strategy is used.

    This method relies on having an accurate source of the camera's angle, such as with an IMU.
    """
    def __init__(self,
                 angle_producer: Callable[[], npt.NDArray[np.float64]],
                 fallback_strategy: Optional[PoseEstimationStrategy] = None,
                 pnp_method: PnPMethod = PnPMethod.IPPE):
        """
        :param angle_producer: A function which returns the most recently measured angle of the world origin in the
                               camera frame. This should be a non-pure function.
        :param fallback_strategy: A strategy to use if only one AprilTag was detected or the PnP solver fails. Cannot
                                  be a :class:`MultiTagSpecialEstimationStrategy`.
        :param pnp_method: A method the strategy will use to solve the Perspective-N-Point problem. Defaults to
                           ``PnPMethod.IPPE``.
        """
        super().__init__()
        if isinstance(fallback_strategy, MultiTagSpecialEstimationStrategy):
            raise TypeError('multitag fallback strategy cannot be another multitag special strategy')
        self.__angle_producer = angle_producer
        self.__fallback_strategy = fallback_strategy
        self.__pnp_method = pnp_method

    @property
    def name(self) -> str:
        return (f'multitag-special-{self.__pnp_method.name}-{self.__fallback_strategy}'
                if self.__fallback_strategy is not None
                else f'multitag-special-{self.__pnp_method.name}')

    def estimate_pose(self, detections: Sequence[AprilTagDetection], field: AprilTagField,
                      camera_params: CameraParameters) -> Optional[Pose]:
        if not detections:
            return None
        if len(detections) == 1:
            return self.__use_fallback_strategy(detections, field, camera_params)

        poses: List[Pose] = []
        actual_rotation_matrix = self.__angle_producer()
        for detection in detections:
            object_points = field.get_corners(detection.tag_id)
            image_points = detection.corners
            try:
                pose_candidates = solve_pnp(object_points, image_points, camera_params, method=self.__pnp_method)
            except EstimationError:
                continue
            poses.append(min(pose_candidates,
                             key=lambda pose: Rotation.from_matrix(pose.rotation_matrix @ actual_rotation_matrix)
                                                      .magnitude()))
        if not poses:
            return self.__use_fallback_strategy(detections, field, camera_params)
        weights = np.array([1 / pose.error for pose in poses])
        translation_vector = np.average(np.hstack([pose.translation_vector for pose in poses]),
                                        axis=1,
                                        weights=weights)
        rotation_matrix = (Rotation.from_matrix(np.array([pose.rotation_matrix for pose in poses]))
                           .mean(weights=weights)
                           .as_matrix())
        return Pose(rotation_matrix=rotation_matrix, translation_vector=translation_vector)

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(angle_producer={self.__angle_producer!r}, '
                f'fallback_strategy={self.__fallback_strategy!r}, '
                f'pnp_method={self.__pnp_method!r})')

    def __use_fallback_strategy(self, detections: Sequence[AprilTagDetection], field: AprilTagField,
                                camera_params: CameraParameters) -> Optional[Pose]:
        if self.__fallback_strategy is None:
            return None
        return self.__fallback_strategy.estimate_pose(detections, field, camera_params)
