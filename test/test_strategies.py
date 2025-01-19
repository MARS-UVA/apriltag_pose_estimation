import json
from collections.abc import Collection
from importlib.resources import files
from itertools import product
from typing import List

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial.transform import Rotation

from apriltag_pose_estimation.apriltag.render import OverlayWriter
from apriltag_pose_estimation.core import AprilTagDetection, AprilTagField, CameraParameters, Pose, PnPMethod
from apriltag_pose_estimation.localization import PoseEstimationStrategy
from apriltag_pose_estimation.localization.strategies import (MultiTagPnPEstimationStrategy,
                                                              LowestAmbiguityEstimationStrategy)
from apriltag_pose_estimation.localization.testcase import PoseEstimationStrategyTestCase

import test.resources


DEPSTECH_CAM_PARAMETERS = CameraParameters(fx=1329.143348,
                                           fy=1326.537785,
                                           cx=945.392392,
                                           cy=521.144703,
                                           k1=-0.348650,
                                           k2=0.098710,
                                           p1=-0.000157,
                                           p2=-0.001851,
                                           k3=0.000000)


def project_points_onto_camera(points: npt.NDArray[np.float64], camera_params: CameraParameters, camera_in_origin: Pose) -> npt.NDArray[np.float64]:
    origin_in_camera = Pose.from_matrix(np.linalg.inv(camera_in_origin.get_matrix()))
    image_points, _ = cv2.projectPoints(points,
                                        origin_in_camera.rotation_vector,
                                        origin_in_camera.translation_vector,
                                        camera_params.get_matrix(),
                                        camera_params.get_distortion_vector())
    return image_points[:, 0, :]


def create_detections(field: AprilTagField, tag_ids: Collection[int], camera_params: CameraParameters, camera_in_origin: Pose) -> List[AprilTagDetection]:
    return [AprilTagDetection(tag_id=tag_id,
                              tag_family=field.tag_family,
                              center=project_points_onto_camera(field[tag_id].translation_vector.reshape((-1, 3)),
                                                                camera_params=camera_params,
                                                                camera_in_origin=camera_in_origin),
                              corners=project_points_onto_camera(field.get_corners(tag_id),
                                                                 camera_params=camera_params,
                                                                 camera_in_origin=camera_in_origin),
                              decision_margin=60,
                              hamming=0,
                              tag_poses=None)
            for tag_id in tag_ids]


fallback_strategies = [
    LowestAmbiguityEstimationStrategy(pnp_method=method)
    for method in PnPMethod
]


strategies = [
    MultiTagPnPEstimationStrategy(fallback_strategy=fallback_strategy, pnp_method=method)
    for fallback_strategy in fallback_strategies
    for method in PnPMethod
    if method is not PnPMethod.IPPE and method is not PnPMethod.AP3P
]


# Note: the AprilTag field was taken from Limelight's model of the FRC 2024 game field.
# https://downloads.limelightvision.io/models/frc2024.fmap
def get_apriltag_field() -> AprilTagField:
    with files(test.resources).joinpath('frc2024.json').open(mode='r') as fp:
        field_data = json.load(fp)
    axis_change = np.array([[0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float64).T
    return AprilTagField(tag_size=0.1651,
                         tag_positions={fiducial['id']: Pose.from_matrix(np.array(fiducial['transform']).reshape((4, 4)) @ axis_change)
                                        for fiducial in field_data['fiducials']},
                         tag_family='tag36h11')


def robot_to_camera_pose(tx: float, ty: float, rot: float, camera_on_robot: Pose) -> Pose:
    axis_change = np.array([[0, 0, 1, 0],
                            [-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float64)
    robot_in_origin = Pose(rotation_matrix=Rotation.from_euler('xyz',
                                                               angles=np.radians([0, 0, rot])).as_matrix(),
                           translation_vector=np.array([tx, ty, 0]).reshape(-1, 1)).get_matrix()
    return Pose.from_matrix(robot_in_origin @ camera_on_robot.get_matrix() @ axis_change)


def get_cases() -> List[PoseEstimationStrategyTestCase]:
    case_id = 0

    def case(actual_camera_pose: Pose, detected_apriltags: Collection[int]) -> PoseEstimationStrategyTestCase:
        nonlocal case_id
        case = PoseEstimationStrategyTestCase(id_=case_id,
                                              actual_camera_pose=actual_camera_pose,
                                              apriltag_field=field,
                                              detected_apriltags=detected_apriltags,
                                              camera_params=DEPSTECH_CAM_PARAMETERS)
        image = np.zeros(shape=(1080, 1920), dtype=np.uint8)
        overlay_writer = OverlayWriter(image, detections=case.detections, camera_params=case.camera_params,
                                       tag_size=case.apriltag_field.tag_size)
        overlay_writer.overlay_square()
        cv2.imwrite(f'case{case_id}.png', image)
        case_id += 1
        return case

    def camera_pose(tx: float, ty: float, rot: float):
        return robot_to_camera_pose(tx, ty, rot, camera_on_robot)

    field = get_apriltag_field()
    camera_on_robot = Pose(
        rotation_matrix=Rotation.from_rotvec(np.radians([0, -30, 0])).as_matrix(),
        translation_vector=np.array([0.304, -0.127, 0.237]).reshape(-1, 1)
    )
    return [
        case(actual_camera_pose=camera_pose(3.5, 0, 5),
             detected_apriltags=[3, 4]),
        case(actual_camera_pose=camera_pose(6.5, 2, 0),
             detected_apriltags=[3, 4]),
        case(actual_camera_pose=camera_pose(6.5, 2.5, 90),
             detected_apriltags=[5]),
    ]


@pytest.mark.parametrize('strategy,case', list(product(strategies, get_cases())))
def test_strategy(strategy: PoseEstimationStrategy, case: PoseEstimationStrategyTestCase):
    pose = strategy.estimate_pose(detections=case.detections, field=case.apriltag_field, camera_params=case.camera_params)
    assert pose is not None
    camera_in_origin = np.linalg.inv(pose.get_matrix())
    assert np.isclose(camera_in_origin, case.actual_camera_pose.get_matrix(), atol=10 ** -2).all()
