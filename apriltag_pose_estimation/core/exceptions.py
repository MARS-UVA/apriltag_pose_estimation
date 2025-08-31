"""Custom exceptions for ``apriltag_pose_estimation``."""


__all__ = ['EstimationError']


class EstimationError(Exception):
    """An error occurred during estimation."""
