"""Provides a class which generates images of AprilTags."""

import ctypes
import os
from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from ...core.bindings import AprilTagFamilyId, AprilTagLibrary, _image_u8_get_array, image_u8, default_search_paths


class AprilTagImageGenerator:
    """A class which generates images of AprilTags."""

    def __init__(self,
                 family: AprilTagFamilyId,
                 search_paths: Sequence[str | os.PathLike] = default_search_paths):
        """
        :param family: The name of the AprilTag family for which images will be generated.
        :param search_paths: Search paths for the C AprilTag library.
        """
        self.__library = AprilTagLibrary(search_paths=search_paths)
        self.__family = self.__library.get_family(family)
        self.__family_name = family

    @property
    def family(self) -> AprilTagFamilyId:
        """The family for which images will be generated."""
        return self.__family_name

    def generate_image(self, tag_id: int, tag_size: int | None = None) -> npt.NDArray[np.uint8]:
        """
        Generate an image for the AprilTag with the given ID in this family at the given size.
        :param tag_id: ID of the AprilTag to generate.
        :param tag_size: Target size of one side of the generated image. If ``None``, the image will be provided at the
           minimum possible size, which is dependent on the family being used.
        :return: A NumPy array containing the image in OpenCV's grayscale format.
        :raise ValueError: If *tag_id* is negative (all valid AprilTag IDs are non-negative integers).
        :raise IndexError: If *tag_id* is out-of-range for the configured family.
        """
        if tag_id < 0:
            raise ValueError('tag_id must be non-negative')
        if tag_id >= self.__family.number_of_codes:
            raise IndexError(f'tag_id out of range for this family (max: {self.__family.number_of_codes - 1})')
        self.__library.libc.apriltag_to_image.restype = ctypes.POINTER(image_u8)
        c_img = self.__library.libc.apriltag_to_image(self.__family.c_ptr, tag_id)
        img = _image_u8_get_array(c_img).copy()
        img = img[:, :img.shape[0]]
        if tag_size is not None:
            img = cv2.resize(img, (tag_size, tag_size), interpolation=cv2.INTER_NEAREST_EXACT)
        return img
