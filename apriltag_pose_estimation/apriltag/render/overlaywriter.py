from collections.abc import Callable, Iterable
from enum import IntEnum
from itertools import chain

import cv2
import numpy as np
import numpy.typing as npt

from .color import Color, RED, GREEN, BLUE
from ...core.euclidean import Transform
from ...core.detection import AprilTagDetection
from ...core.camera import CameraParameters


__all__ = ['Font', 'OverlayWriter']


class Font(IntEnum):
    """A font that the OverlayWriter can use."""
    HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
    HERSHEY_PLAIN = cv2.FONT_HERSHEY_PLAIN
    HERSHEY_DUPLEX = cv2.FONT_HERSHEY_DUPLEX
    HERSHEY_COMPLEX = cv2.FONT_HERSHEY_COMPLEX
    HERSHEY_TRIPLEX = cv2.FONT_HERSHEY_TRIPLEX
    HERSHEY_COMPLEX_SMALL = cv2.FONT_HERSHEY_COMPLEX_SMALL
    HERSHEY_SCRIPT_SIMPLEX = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    HERSHEY_SCRIPT_COMPLEX = cv2.FONT_HERSHEY_SCRIPT_COMPLEX


def _default_label_func(detection: AprilTagDetection) -> str:
    return f'ID: {detection.tag_id}'


class OverlayWriter:
    """An OverlayWriter overlays graphics onto an image which was used to detect AprilTags."""
    def __init__(self,
                 image: npt.NDArray[np.uint8],
                 detections: Iterable[AprilTagDetection],
                 camera_params: CameraParameters,
                 tag_size: float):
        """
        Initializes a new OverlayWriter.
        :param image: The image on which to draw the overlay.
        :param detections: The list of AprilTags for which overlay data will be drawn.
        :param camera_params: The parameters of the camera used to take the image.
        :param tag_size: The size of the AprilTags, in meters.
        """
        self.__image = image
        self.__detections = detections
        self.__camera_params = camera_params
        self.__tag_size = tag_size

    def overlay_square(self, thickness: int = 2, color: Color = BLUE, show_corners: bool = False,
                       corner_point_radius: int = 8) -> None:
        """
        Overlay the square corresponding to each detected AprilTag on the image.
        :param thickness: The thickness of the lines for the square's edges (default: 2).
        :param color: The color of the square (default: blue).
        :param show_corners: Whether to display the corner points of the square (default: ``False``).
        :param corner_point_radius: The radius of the corner points of the square (default: 8). Has no effect if
                                    ``show_corners`` is ``False``.
        """
        for detection in self.__detections:
            points = detection.corners.astype(np.int_)
            for src, dest in zip(points[:4], chain(points[1:4], points[:1])):
                cv2.line(self.__image, src, dest, color=color.bgr(), thickness=thickness)
            if show_corners:
                for point, pt_color in zip(points, [Color(red=0x00, green=0x00, blue=0xff),
                                                    Color(red=0xff, green=0x00, blue=0x80),
                                                    Color(red=0xff, green=0xff, blue=0x00),
                                                    Color(red=0x00, green=0xff, blue=0x7f)]):
                    cv2.circle(self.__image, point, corner_point_radius, color=pt_color.bgr(), thickness=-1)

    def overlay_label(self,
                      label_func: Callable[[AprilTagDetection], str] = _default_label_func,
                      font: Font = Font.HERSHEY_SIMPLEX,
                      scale: float = 1.0,
                      thickness: int = 2,
                      color: Color = BLUE) -> None:
        """
        Overlay a label for each tag onto the image.
        :param label_func: A function which takes an AprilTagDetection and returns a label. Defaults to a function which
                           returns a label indicating the ID (e.g., for tag 0, returns "ID: 0").
        :param font: The font to use for the label (default: Hershey Simplex).
        :param scale: The scale factor of the label (default: 1.0).
        :param thickness: The thickness of the label (default: 2).
        :param color: The color of the label (default: blue).
        """
        for detection in self.__detections:
            cv2.putText(self.__image,
                        label_func(detection),
                        np.round(detection.center).astype(np.int_),
                        int(font),
                        scale,
                        color.bgr(),
                        thickness,
                        cv2.LINE_AA)

    def overlay_axes(self,
                     axis_length: float = 0.5,
                     thickness: int = 2,
                     invert_x: bool = False,
                     invert_y: bool = False,
                     invert_z: bool = False,
                     x_color: Color = RED,
                     y_color: Color = GREEN,
                     z_color: Color = BLUE) -> None:
        """
        Overlay the axes corresponding to each detected AprilTag on the image.
        :param axis_length: The length of each axis as a proportion of the tag size (default: 0.5).
        :param thickness: The thickness of the lines for the axes (default: 2).
        :param invert_x: Whether to invert the x-axis (default: False).
        :param invert_y: Whether to invert the y-axis (default: False).
        :param invert_z: Whether to invert the z-axis (default: False).
        :param x_color: The color with which to draw the x-axis (default: red).
        :param y_color: The color with which to draw the y-axis (default: green).
        :param z_color: The color with which to draw the z-axis (default: blue).
        """
        object_points = np.array([[
            [0, 0, 0],
            [-1 if invert_x else 1, 0, 0],
            [0, -1 if invert_y else 1, 0],
            [0, 0, -1 if invert_z else 1],
        ]], dtype=np.float64) * self.__tag_size * axis_length
        for detection in self.__detections:
            projected_points = self.__project(object_points, detection.best_tag_pose)
            cv2.line(self.__image, projected_points[0], projected_points[3], color=z_color.bgr(), thickness=thickness)
            cv2.line(self.__image, projected_points[0], projected_points[1], color=x_color.bgr(), thickness=thickness)
            cv2.line(self.__image, projected_points[0], projected_points[2], color=y_color.bgr(), thickness=thickness)

    def overlay_cube(self, color: Color = RED) -> None:
        """
        Overlay the cube corresponding to each detected AprilTag on the image.
        :param color: The color with which to draw the cube (default: red).
        """
        object_points = np.array([[
            [-1, -1, 0],
            [+1, -1, 0],
            [+1, +1, 0],
            [-1, +1, 0],
            [-1, -1, -2],
            [+1, -1, -2],
            [+1, +1, -2],
            [-1, +1, -2],
            [0, 0, -1]
        ]]) / 2 * self.__tag_size
        edges = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7]
        ])
        for detection in self.__detections:
            projected_points = self.__project(object_points, detection.best_tag_pose)
            for i, j in edges:
                cv2.line(self.__image, projected_points[i], projected_points[j], color=color.bgr(), thickness=2)

    def overlay_cubes(self, color: Color = RED) -> None:
        """
        Overlay cubes corresponding to each possible pose of the detected AprilTags on the image. Poses with higher
        reprojection errors are drawn with opacity.
        :param color: The color with which to draw the cubes (default: red).
        """
        object_points = np.array([[
            [-1, -1, 0],
            [+1, -1, 0],
            [+1, +1, 0],
            [-1, +1, 0],
            [-1, -1, -2],
            [+1, -1, -2],
            [+1, +1, -2],
            [-1, +1, -2],
            [0, 0, -1]
        ]]) / 2 * self.__tag_size
        edges = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7]
        ])
        for detection in self.__detections:
            total_errors = sum(pose.error if pose.error else 0 for pose in detection.tag_poses)
            for pose in detection.tag_poses:
                projected_points = self.__project(object_points, pose)
                if pose.error is None or pose.error == 0 or pose.error == total_errors:
                    alpha = 1.
                else:
                    alpha = 1 - (pose.error / total_errors)
                image_copy = np.array(self.__image)
                for i, j in edges:
                    cv2.line(self.__image, projected_points[i], projected_points[j], color=color.bgr(), thickness=2)
                cv2.addWeighted(self.__image, alpha, image_copy, 1 - alpha, 0, self.__image)

    def __project(self, object_points: npt.NDArray[np.float64], pose: Transform):
        projected_points, _ = cv2.projectPoints(object_points,
                                                pose.opencv_rotation_vector,
                                                pose.opencv_translation_vector,
                                                self.__camera_params.get_matrix(),
                                                self.__camera_params.get_distortion_vector())
        projected_points = np.round(projected_points).astype(int).reshape((-1, 2))
        return np.round(projected_points).astype(int).reshape((-1, 2))
