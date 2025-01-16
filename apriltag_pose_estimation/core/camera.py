from dataclasses import dataclass

import numpy as np
from numpy import typing as npt


@dataclass(frozen=True, kw_only=True)
class CameraParameters:
    """Parameters that represent characteristic information of a camera."""
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
        """Create a CameraParameters object from a camera matrix and distortion vector."""
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
        """Returns a camera matrix created from the camera parameters."""
        camera_matrix = np.zeros((3, 3), dtype=np.float32)
        camera_matrix[0, 0] = self.fx
        camera_matrix[1, 1] = self.fy
        camera_matrix[0, 2] = self.cx
        camera_matrix[1, 2] = self.cy
        camera_matrix[2, 2] = 1
        return camera_matrix

    def get_distortion_vector(self) -> npt.NDArray[np.float32]:
        """Returns a distortion vector created from the camera parameters."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)
