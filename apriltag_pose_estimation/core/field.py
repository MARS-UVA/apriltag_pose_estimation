from collections.abc import Mapping, KeysView

from .euclidean import Pose


class Field(Mapping[int, Pose]):
    __slots__ = '__tag_size', '__tag_positions', '__tag_family'

    def __init__(self, tag_size: float, tag_positions: dict[int, Pose], tag_family: str = 'tag36h11'):
        self.__tag_size = tag_size
        self.__tag_positions = tag_positions
        self.__tag_family = tag_family

    @property
    def tag_size(self) -> float:
        return self.__tag_size

    @property
    def tag_family(self) -> str:
        return self.__tag_family

    def __getitem__(self, __key: int):
        return self.__tag_positions[__key]

    def __len__(self):
        return len(self.__tag_positions)

    def __iter__(self):
        return iter(self.__tag_positions)

    def tag_ids(self) -> KeysView[int]:
        return self.__tag_positions.keys()
