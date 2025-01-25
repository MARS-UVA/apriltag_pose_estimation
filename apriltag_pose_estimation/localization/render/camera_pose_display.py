import json
from importlib.resources import files
from typing import TextIO, Optional

import cv2
import numpy as np
from PyQt5 import Qt
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from scipy.spatial.transform import Rotation

from apriltag_pose_estimation.apriltag.render import OverlayWriter
from apriltag_pose_estimation.core import CameraParameters, PnPMethod
from apriltag_pose_estimation.localization import PoseEstimator
from apriltag_pose_estimation.localization.strategies import MultiTagPnPEstimationStrategy, \
    LowestAmbiguityEstimationStrategy
from apriltag_pose_estimation.core import Transform
from apriltag_pose_estimation.core.field import AprilTagField
from apriltag_pose_estimation.localization.render import resource


LOGITECH_CAM_PARAMETERS = CameraParameters(fx=1303.4858439074037,
                                           fy=1313.7268166341282,
                                           cx=953.0550046450967,
                                           cy=487.428417101308,
                                           k1=-0.04617273252027395,
                                           k2=0.2753447226702122,
                                           p1=-0.010067837101803492,
                                           p2=-0.005296327017158184,
                                           k3=-0.38168395944619604)
"""Camera parameters for a Logitech C920 webcam."""


FACETIME_CAM_PARAMETERS = CameraParameters(fx=954.0874994019651,
                                           fy=949.9159862376827,
                                           cx=660.572082940535,
                                           cy=329.78814306885795,
                                           k1=329.78814306885795,
                                           k2=-0.13207581435883872,
                                           p1=-0.000045055253893522236,
                                           p2=-0.007745497656725853,
                                           k3=0.11519181871308336)
"""Camera parameters for a FaceTime HD camera."""


class CameraPoseDisplay:
    def __init__(self, field: AprilTagField):
        plane_scale_factor = 9 / 5  # currently specific to the present tag standard
        self.__plotter = BackgroundPlotter()
        for tag_id, tag_pose in field.items():
            tag_pose_standard = tag_pose
            texture = pv.read_texture(str(files(resource).joinpath(f'{tag_id}.png')))
            mesh: pv.PolyData = pv.Plane(center=tag_pose_standard.translation,
                                         direction=tag_pose_standard.rotation.as_matrix()[:, 2],
                                         i_size=field.tag_size * plane_scale_factor,
                                         j_size=field.tag_size * plane_scale_factor)
            mesh.point_data.clear()
            mesh.texture_map_to_plane(inplace=True)
            self.__plotter.add_mesh(mesh, texture=texture)
        mesh_path = str(files(resource).joinpath('camera.stl'))

        self.__original_camera_mesh: pv.DataSet = pv.read_meshio(mesh_path)
        self.__displayed_camera_mesh: pv.DataSet = self.__original_camera_mesh.copy(deep=True)
        self.__camera_mesh_actor = self.__plotter.add_mesh(self.__displayed_camera_mesh)
        self.__camera_mesh_actor.SetVisibility(True)

        self.__plotter.camera_position = 'xy'

    @property
    def plotter(self) -> BackgroundPlotter:
        return self.__plotter

    def update(self, origin_in_camera: Optional[Transform] = None) -> None:
        if origin_in_camera is not None:
            camera_in_origin = origin_in_camera.inv()
            self.__displayed_camera_mesh.deep_copy(self.__original_camera_mesh.transform(camera_in_origin.matrix.astype(float), inplace=False))
        self.__plotter.update()

    def close(self) -> None:
        self.__plotter.close()


def get_field(fp: TextIO) -> AprilTagField:
    field_dict = json.load(fp)
    return AprilTagField(tag_size=field_dict['tag_size'],
                         tag_family=field_dict['tag_family'],
                         tag_positions={tag_data['id']: Transform.make(rotation=Rotation.from_rotvec(tag_data['rotation_vector']),
                                                                       translation=tag_data['translation_vector'],
                                                                       input_space='tag_optical',
                                                                       output_space='world')
                                        for tag_data in field_dict['fiducials']})


def get_basic_field() -> AprilTagField:
    optical_to_standard = Transform.from_matrix(np.array([[0, 0, 1, 0],
                                                          [-1, 0, 0, 0],
                                                          [0, -1, 0, 0],
                                                          [0, 0, 0, 1]]),
                                                input_space='world_optical',
                                                output_space='world')
    # Tag size is measured as the side length of the interior white square in meters
    return AprilTagField(tag_size=0.080,
                         tag_family='tagStandard41h12',
                         tag_positions={
                             0: optical_to_standard @ Transform.identity(input_space='tag_optical',
                                                                         output_space='world_optical'),
                             2: optical_to_standard @ Transform.make(rotation=Rotation.identity(),
                                                                     translation=[-0.83, 0, 0],
                                                                     input_space='tag_optical',
                                                                     output_space='world_optical'),
                          })


def main() -> None:
    def timer_callback() -> None:
        not_closed, frame = video_capture.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = estimator.estimate_pose(image)
        if result.estimated_pose is not None:
            display.update(result.estimated_pose)
        overlay_writer = OverlayWriter(frame,
                                       detections=result.detections,
                                       camera_params=estimator.camera_params,
                                       tag_size=estimator.field.tag_size)
        overlay_writer.overlay_square(show_corners=True)
        overlay_writer.overlay_label()
        cv2.imshow('camera', frame)

    estimator = PoseEstimator(
        strategy=MultiTagPnPEstimationStrategy(fallback_strategy=LowestAmbiguityEstimationStrategy()),
        field=get_basic_field(),
        camera_params=LOGITECH_CAM_PARAMETERS,
        nthreads=2,
        quad_sigma=0,
        refine_edges=1,
        decode_sharpening=0.25
    )

    video_capture = cv2.VideoCapture(1)

    cv2.namedWindow('camera')

    display = CameraPoseDisplay(estimator.field)

    timer = Qt.QTimer(display.plotter.app_window)
    timer.setSingleShot(False)
    timer.setInterval(1000 // 24)
    timer.timeout.connect(timer_callback)
    timer.start()

    try:
        display.plotter.app.exec()
    finally:
        cv2.destroyWindow('camera')


if __name__ == '__main__':
    main()
