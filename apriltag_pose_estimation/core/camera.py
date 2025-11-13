"""
This module provides the :py:class:`CameraParameters` class for representing lens distortion, as well as instances for
cameras that the MARS team uses for testing purposes.

Avoid using the built-in instances, since calibrating your camera yourself will be more accurate. The
:py:mod:`apriltag_pose_estimation.calibrate` provides a tool for calibrating your camera.
"""


from dataclasses import dataclass

import numpy as np
from numpy import typing as npt


__all__ = ['CameraParameters',
           'DEPSTECH_CAM_PARAMETERS',
           'LOGITECH_CAM_PARAMETERS',
           'IPHONE_13_MINI_MAIN_CAM_PARAMETERS',
           'ARDUCAM_OV9281_PARAMETERS',
           'GALAXY_S24_PLUS_MAIN_CAM_PARAMETERS']


@dataclass(frozen=True, kw_only=True)
class CameraParameters:
    """
    Parameters that represent characteristic information of a camera.

    When a camera is calibrated with the :py:mod:`apriltag_pose_estimation.calibrate` tool, OpenCV's calibration
    API, or similar tools, a set of camera parameters which describe the distortion of the camera is returned. These
    parameters are necessary for performing 3D tracking.

    This is a frozen dataclass, so its instances are immutable.
    """
    fx: float
    """Focal length in the x direction."""
    fy: float
    """Focal length in the y direction."""
    cx: float
    """X coordinate of the optical center."""
    cy: float
    """Y coordinate of the optical center."""
    k1: float
    """First parameter of radial distortion."""
    k2: float
    """Second parameter of radial distortion."""
    p1: float
    """First parameter of tangential distortion."""
    p2: float
    """Second parameter of tangential distortion."""
    k3: float
    """Third parameter of radial distortion."""

    @classmethod
    def from_matrices(cls, camera_matrix: npt.NDArray[np.float64], distortion_vector: npt.NDArray[np.float64]):
        """
        Create a ``CameraParameters`` object from a camera matrix and distortion vector.

        This method is useful for converting results from an OpenCV calibration to a ``CameraParameters`` object. The
        expected order of values in the camera is::

           | fx  0 cx |
           |  0 fy cy |
           |  0  0  1 |

        And the expected order of values in the distortion vector is::

           | k1 k2 p1 p2 k3 |

        This matches the matrices returned by OpenCV's ``calibrateCamera()`` function.

        :param camera_matrix: The intrinsic camera matrix for the camera, likely returned from OpenCV's
           ``calibrateCamera()`` function.
        :param distortion_vector: The distortion vector for the camera, likely returned from OpenCV's
           ``calibrateCamera()`` function.
        :return: A new CameraParameters object representing the camera's distortion.
        """
        return cls(
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]),
            cy=float(camera_matrix[1, 2]),
            k1=float(distortion_vector[0]),
            k2=float(distortion_vector[1]),
            p1=float(distortion_vector[2]),
            p2=float(distortion_vector[3]),
            k3=float(distortion_vector[4]),
        )

    def get_matrix(self) -> npt.NDArray[np.float32]:
        """
        Returns a camera matrix created from the camera parameters.

        The returned matrix has the following format::

           | fx  0 cx |
           |  0 fy cy |
           |  0  0  1 |
        """
        camera_matrix = np.zeros((3, 3), dtype=np.float32)
        camera_matrix[0, 0] = self.fx
        camera_matrix[1, 1] = self.fy
        camera_matrix[0, 2] = self.cx
        camera_matrix[1, 2] = self.cy
        camera_matrix[2, 2] = 1
        return camera_matrix

    def get_distortion_vector(self) -> npt.NDArray[np.float32]:
        """
        Returns a distortion vector created from the camera parameters.

        The returned vector has the following format::

           | k1 k2 p1 p2 k3 |
        """
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)


DEPSTECH_CAM_PARAMETERS = CameraParameters(fx=1329.143348,
                                           fy=1326.537785,
                                           cx=945.392392,
                                           cy=521.144703,
                                           k1=-0.348650,
                                           k2=0.098710,
                                           p1=-0.000157,
                                           p2=-0.001851,
                                           k3=0.000000)
"""Camera parameters for a Depstech webcam."""

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

IPHONE_13_MINI_MAIN_CAM_PARAMETERS = CameraParameters(fx=975.0944324169702,
                                                      fy=975.7210449202735,
                                                      cx=651.2516360655774,
                                                      cy=354.20133381181955,
                                                      k1=0.16202747407118365,
                                                      k2=-0.8339518338379913,
                                                      p1=-0.0021712184519895234,
                                                      p2=0.003869488985826069,
                                                      k3=1.326544376051906)
"""Camera parameters for the iPhone 13 mini's main back camera."""

ARDUCAM_OV9281_PARAMETERS = CameraParameters(fx=921.4459062879504, fy=920.5545512136466, cx=640.4116071984264, cy=389.86971089068913, k1=0.07166453170075777, k2=-0.11227733992947839, p1=-0.0014616393622597851, p2=-0.0014945237103234227, k3=0.04562948613626775)
"""Camera parameters for the ArduCam OV9281 camera module."""


FACETIME_HD_CAMERA_PARAMETERS = CameraParameters(fx=954.0874994019651,
                                                 fy=949.9159862376827,
                                                 cx=660.572082940535,
                                                 cy=329.78814306885795,
                                                 k1=0.036998502838994515,
                                                 k2=-0.13207581435883872,
                                                 p1=-0.000045055253893522236,
                                                 p2=-0.007745497656725853,
                                                 k3=0.11519181871308336)

GALAXY_S24_PLUS_MAIN_CAM_PARAMETERS = CameraParameters(fx=1577.6710882155696,
                                                      fy=1578.1918913274176,
                                                      cx=536.7027520891314,
                                                      cy=980.2679188012452,
                                                      k1=-0.0645156141142605,
                                                      k2=0.14588713723707655,
                                                      p1=0.003958174370212018,
                                                      p2=0.00012977471972707647,
                                                      k3=-0.1784624986145353)
"""Camera parameters for the Samsung Galaxy S24 Plus main camera."""
