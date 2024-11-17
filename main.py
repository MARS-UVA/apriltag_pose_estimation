import cv2
import numpy as np

from apriltag_pose_estimation import AprilTagPoseEstimator, CameraParameters
from apriltag_pose_estimation.strategies import (HomographyOrthogonalIterationStrategy, PerspectiveNPointStrategy,
                                                 PnPMethod)
from apriltag_pose_estimation.render import OverlayWriter


DEPSTECH_CAM_PARAMETERS = CameraParameters(fx=1329.143348,
                                           fy=1326.537785,
                                           cx=945.392392,
                                           cy=521.144703,
                                           k1=-0.348650,
                                           k2=0.098710,
                                           p1=-0.000157,
                                           p2=-0.001851,
                                           k3=0.000000)


LOGITECH_CAM_PARAMETERS = CameraParameters(fx=1394.6027293299926,
                                           fy=1394.6027293299926,
                                           cx=995.588675691456,
                                           cy=599.3212928484164,
                                           k1=0.11480806073904032,
                                           k2=-0.21946985653851792,
                                           p1=0.0012002116999769957,
                                           p2=0.008564577708855225,
                                           k3=0.11274677130853494)


def main() -> None:
    estimator = AprilTagPoseEstimator(strategy=PerspectiveNPointStrategy(method=PnPMethod.IPPE),
                                      tag_size=0.150,
                                      camera_params=LOGITECH_CAM_PARAMETERS,
                                      families='tagStandard41h12',
                                      nthreads=2,
                                      quad_sigma=0,
                                      refine_edges=1,
                                      decode_sharpening=0.25)
    cv2.namedWindow('Capture')
    capture = cv2.VideoCapture(0)
    not_closed = True
    while not_closed:
        not_closed, frame = capture.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if estimates := estimator.estimate_tag_pose(image):
            for estimate in estimates:
                print(f'Distance to tag {estimate.tag_id} from camera: '
                      f'{np.linalg.norm(estimate.tag_pose.translation_vector.T)} m')
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            overlay_writer = OverlayWriter(image, estimates, LOGITECH_CAM_PARAMETERS, tag_size=0.150)
            overlay_writer.overlay_label()
            overlay_writer.overlay_cube()
        cv2.imshow('Capture', image)
        key = cv2.waitKey(20)
        if key == 27:
            break

    cv2.destroyWindow('Capture')


if __name__ == '__main__':
    main()
