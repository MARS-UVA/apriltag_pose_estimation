import datetime
from pathlib import Path

import cv2
import csv

from apriltag_pose_estimation.core import ARDUCAM_OV9281_PARAMETERS

from apriltag_pose_estimation.core.camera import FACETIME_HD_CAMERA_PARAMETERS
from apriltag_pose_estimation.localization import CameraLocalizer, load_field
from apriltag_pose_estimation.localization.strategies import MultiTagPnPStrategy, \
    LowestAmbiguityStrategy


def main() -> None:
    examples_path = Path(__file__).parent

    estimators = []
    for i in range(3):
        with (examples_path / f'mars_field{i + 1}.json').open(mode='r') as f:
            field = load_field(f)

        estimators.append(CameraLocalizer(
            strategy=MultiTagPnPStrategy(fallback_strategy=LowestAmbiguityStrategy()),
            field=field,
            camera_params=ARDUCAM_OV9281_PARAMETERS,
            nthreads=8,
            quad_sigma=0,
            refine_edges=True,
            decode_sharpening=0.25,
        ))

    video_capture = cv2.VideoCapture(0)

    cv2.namedWindow('camera')

    try:
        with open(f'localization_data_{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}.csv', mode='w+', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['timestamp', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3'])
            while True:
                not_closed, frame = video_capture.read()
                now = datetime.datetime.now()
                if not not_closed:
                    return
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                results = [estimator.estimate_pose(img_gray) for estimator in estimators]
                if all(result.estimated_pose is not None for result in results):
                    writer.writerow(map(str, [now.timestamp(),
                                              *results[0].estimated_pose.translation,
                                              *results[1].estimated_pose.translation,
                                              *results[2].estimated_pose.translation,]))

                cv2.imshow('camera', frame)
                cv2.waitKey(1)
    finally:
        cv2.destroyWindow('camera')


if __name__ == '__main__':
    main()
