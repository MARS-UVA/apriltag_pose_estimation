import datetime
from pathlib import Path

import cv2
import csv

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
        nthreads=8,
        quad_sigma=0,
        refine_edges=True,
        decode_sharpening=0.25,
    )


    video_capture = cv2.VideoCapture(0)

    cv2.namedWindow('camera')

    try:
        with open(f'localization_data_{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}.csv', mode='w+') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['timestamp', 'x', 'y', 'z', 'x_rot', 'y_rot', 'z_rot'])
            while True:
                not_closed, frame = video_capture.read()
                now = datetime.datetime.now()
                if not not_closed:
                    return
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                results = estimator.estimate_pose(img_gray)
                if results.estimated_pose is not None:
                    writer.writerow(map(str, [now.timestamp(),
                                              *results.estimated_pose.translation,
                                              *results.estimated_pose.rotation.as_euler(seq='xyz', degrees=True)]))

                cv2.imshow('camera', frame)
                cv2.waitKey(1)
    finally:
        cv2.destroyWindow('camera')


if __name__ == '__main__':
    main()
