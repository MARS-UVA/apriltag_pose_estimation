#!/usr/bin/env python3

import cv2

from apriltag_pose_estimation.apriltag import AprilTagPoseEstimator
from apriltag_pose_estimation.apriltag.render import OverlayWriter
from apriltag_pose_estimation.apriltag.strategies import PerspectiveNPointStrategy
from apriltag_pose_estimation.core import PnPMethod
from apriltag_pose_estimation.core.camera import LOGITECH_CAM_PARAMETERS


def main() -> None:
    estimator = AprilTagPoseEstimator(
        strategy=PerspectiveNPointStrategy(method=PnPMethod.ITERATIVE),
        tag_size=0.100,
        camera_params=LOGITECH_CAM_PARAMETERS,
        nthreads=2,
        quad_sigma=0,
        refine_edges=True,
        decode_sharpening=0.25,
    )

    video_capture = cv2.VideoCapture(0)

    cv2.namedWindow('camera')

    try:
        while True:
            not_closed, frame = video_capture.read()
            if not not_closed:
                return
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = estimator.estimate_tag_pose(img_gray)
            overlay_writer = OverlayWriter(frame, results, camera_params=LOGITECH_CAM_PARAMETERS, tag_size=0.100)
            overlay_writer.overlay_cubes()
            overlay_writer.overlay_label(scale=3, thickness=5)
            cv2.imshow('camera', frame)
            cv2.waitKey(1)
    finally:
        cv2.destroyWindow('camera')


if __name__ == '__main__':
    main()
