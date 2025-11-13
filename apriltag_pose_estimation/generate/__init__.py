"""
This package is an executable package for generating printable AprilTags at specific sizes.

The package also exposes the primary function for generating PDFs.
"""

from .tagset import *

__all__ = [
    'generate_apriltag_pdf'
]
