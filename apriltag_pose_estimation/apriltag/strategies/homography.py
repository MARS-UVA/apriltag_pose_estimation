from typing import List

import numpy as np
import numpy.typing as npt
from ..estimation import AprilTagPoseEstimationStrategy
from ...core.camera import CameraParameters
from ...core.detection import AprilTagDetection, AprilTagDetector


__all__ = ['HomographyOrthogonalIterationStrategy']


class HomographyOrthogonalIterationStrategy(AprilTagPoseEstimationStrategy):
    """
    A pose estimation strategy which uses Orthogonal Iteration on a homography-based estimate.

    This uses the homography method described in [1] for the initial estimate. Then Orthogonal Iteration [2] is used to
    refine this estimate. Then [3] is used to find a potential second local minima and Orthogonal Iteration is used to
    refine this second estimate.

    [1]: E. Olson, "Apriltag: A robust and flexible visual fiducial system," in
         2011 IEEE International Conference on Robotics and Automation,
         May 2011, pp. 3400-3407.
    [2]: Lu, G. D. Hager and E. Mjolsness, "Fast and globally convergent pose
         estimation from video images," in IEEE Transactions on Pattern Analysis
         and Machine Intelligence, vol. 22, no. 6, pp. 610-622, June 2000.
         doi: 10.1109/34.862199
    [3]: Schweighofer and A. Pinz, "Robust Pose Estimation from a Planar Target,"
         in IEEE Transactions on Pattern Analysis and Machine Intelligence,
         vol. 28, no. 12, pp. 2024-2030, Dec. 2006.  doi: 10.1109/TPAMI.2006.252
    """
    def estimate_tag_pose(self,
                          image: npt.NDArray[np.uint8],
                          detector: AprilTagDetector,
                          camera_params: CameraParameters,
                          tag_size: float) -> List[AprilTagDetection]:
        return detector.detect(image,
                               estimate_tag_pose=True,
                               camera_params=camera_params,
                               tag_size=tag_size)

    @property
    def name(self):
        return 'homography'
