from collections.abc import Mapping, KeysView
from typing import Dict

import numpy as np
import numpy.typing as npt

from .euclidean import Pose


__all__ = ['AprilTagField']


class AprilTagField(Mapping[int, Pose]):
    __slots__ = '__tag_size', '__tag_positions', '__tag_family'

    def __init__(self, tag_size: float, tag_positions: dict[int, Pose], tag_family: str = 'tag36h11'):
        self.__tag_size = tag_size
        self.__tag_positions = tag_positions
        self.__tag_family = tag_family
        self.__corners: Dict[int, npt.NDArray[np.float64]] = {}

        self.__calculate_corners()

    @property
    def tag_size(self) -> float:
        return self.__tag_size

    @property
    def tag_family(self) -> str:
        return self.__tag_family

    def __getitem__(self, __key: int):
        return self.__tag_positions[__key]

    def get_corners(self, *tag_ids: int) -> npt.NDArray[np.float64]:
        if not tag_ids:
            return np.vstack([corners for corners in self.__corners.values()])
        return np.vstack([corners for tag_id, corners in self.__corners.items() if tag_id in tag_ids])

    def __len__(self):
        return len(self.__tag_positions)

    def __iter__(self):
        return iter(self.__tag_positions)

    def tag_ids(self) -> KeysView[int]:
        return self.__tag_positions.keys()

    def __calculate_corners(self) -> None:
        corner_points = np.array([
            [-1, +1, 0, 1],
            [+1, +1, 0, 1],
            [+1, -1, 0, 1],
            [-1, -1, 0, 1],
        ]) / 2 * self.tag_size
        self.__corners = {tag_id: pose.get_matrix() @ corner_points for tag_id, pose in self.__tag_positions.items()}
