#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2

from apriltag_pose_estimation.apriltag.render import OverlayWriter, MAGENTA
from apriltag_pose_estimation.core import AprilTagDetector


def _main() -> None:
    parser = argparse.ArgumentParser(description='Detect AprilTags in an image')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--image', type=Path,
                       help='The image in which AprilTags will be detected')
    group.add_argument('-c', '--camera', type=int,
                       help='The OpenCV video number index. On Unix systems, this usually corresponds to the number '
                            'of the corresponding video device file (/dev/video*).')
    args = parser.parse_args()

    detector = AprilTagDetector()
    if args.image is not None:
        img_color = cv2.imread(str(args.image))
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(img_gray)
        overlay_writer = OverlayWriter(img_color, detections)
        overlay_writer.overlay_square(color=MAGENTA, thickness=5)
        overlay_writer.overlay_label(color=MAGENTA, scale=3, thickness=5)
        cv2.imwrite(str(args.image.with_stem(args.image.stem + '_detections')), img_color)
    elif args.camera is not None:
        video_capture = cv2.VideoCapture(args.camera)
        cv2.namedWindow('camera')
        try:
            while True:
                not_closed, frame = video_capture.read()
                if not not_closed:
                    return
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(img_gray)
                overlay_writer = OverlayWriter(frame, detections)
                overlay_writer.overlay_square(color=MAGENTA, thickness=5)
                overlay_writer.overlay_label(color=MAGENTA, scale=3, thickness=5)
                cv2.imshow('camera', frame)
                cv2.waitKey(1)
        finally:
            cv2.destroyWindow('camera')
    else:
        raise RuntimeError('invalid args were processed')


if __name__ == '__main__':
    _main()
