from pathlib import Path

import cv2

from apriltag_pose_estimation.core.camera import FACETIME_HD_CAMERA_PARAMETERS
from apriltag_pose_estimation.localization import CameraLocalizer, load_field
from apriltag_pose_estimation.localization.strategies import MultiTagPnPStrategy, \
    LowestAmbiguityStrategy


def main() -> None:
    examples_path = Path(__file__).parent
    with (examples_path / 'onetag_testfield.json').open(mode='r') as f:
        field = load_field(f)

    estimator = CameraLocalizer(
        strategy=MultiTagPnPStrategy(fallback_strategy=LowestAmbiguityStrategy()),
        field=field,
        camera_params=FACETIME_HD_CAMERA_PARAMETERS,
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

            results = estimator.estimate_pose(img_gray)
            if results.estimated_pose is not None:
                print(results)
            cv2.imshow('camera', frame)
            cv2.waitKey(1)
    finally:
        cv2.destroyWindow('camera')


if __name__ == '__main__':
    main()
