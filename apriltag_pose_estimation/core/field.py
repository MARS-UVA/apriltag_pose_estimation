from collections.abc import Mapping
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from .euclidean import Pose


__all__ = ['AprilTagField']


class AprilTagField(Mapping[int, Pose]):
    """
    A class whose instances store information about the AprilTags in the region in which the robot is operating.

    This class implements the :class:`Mapping` protocol, where the keys are tag IDs and the values are their poses in
    the world frame.
    """

    __slots__ = '__tag_size', '__tag_positions', '__tag_family', '__corners'

    def __init__(self, tag_size: float, tag_positions: dict[int, Pose], tag_family: str = 'tag36h11'):
        """
        :param tag_size: The size of the AprilTags on the field in meters.
        :param tag_positions: A dictionary from the IDs of tags on the field to their poses in the world frame.
        :param tag_family: The AprilTag family of which the tags on the field are a part.
        """
        self.__tag_size = tag_size
        self.__tag_positions = tag_positions
        self.__tag_family = tag_family
        self.__corners: Dict[int, npt.NDArray[np.float64]] = {}

        self.__calculate_corners()

    @property
    def tag_size(self) -> float:
        """The size of the AprilTags on the field in meters."""
        return self.__tag_size

    @property
    def tag_family(self) -> str:
        """The AprilTag family of which the tags on the field are a part."""
        return self.__tag_family

    def __getitem__(self, __key: int):
        return self.__tag_positions[__key]

    def get_corners(self, *tag_ids: int) -> npt.NDArray[np.float64]:
        """
        Returns corner points of the AprilTags with the given IDs in the world frame.

        The corner points are returned in a 4nx3 array in the same order the IDs were given.
        :param tag_ids: The IDs of the AprilTags for which corner points will be retrieved.
        :return: A 4nx3 array containing the corner points of the AprilTags.
        """
        if not tag_ids:
            return np.zeros(shape=(0, 4))
        return np.vstack([corners for tag_id, corners in self.__corners.items() if tag_id in tag_ids])

    def __len__(self):
        return len(self.__tag_positions)

    def __iter__(self):
        return iter(self.__tag_positions)

    def get_tag_ids(self) -> List[int]:
        """Returns a list of IDs corresponding to the AprilTags on this field in no particular order."""
        return list(self.__tag_positions.keys())

    def __calculate_corners(self) -> None:
        corner_points = np.array([
            [-1, +1, 0],
            [+1, +1, 0],
            [+1, -1, 0],
            [-1, -1, 0],
        ]) / 2 * self.tag_size
        corner_points = np.hstack([corner_points, np.ones((4, 1))])
        self.__corners = {tag_id: (pose.get_matrix() @ corner_points.T)[:-1, :].T for tag_id, pose in self.__tag_positions.items()}
