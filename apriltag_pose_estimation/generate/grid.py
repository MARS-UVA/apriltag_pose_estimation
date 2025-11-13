"""Provides a function for generating PDF documents with AprilTags for printing with a traditional printer."""
from collections import deque
from collections.abc import Iterable
from itertools import product, chain, count
from pathlib import Path
from typing import Literal

import numpy as np

from ..core.bindings import AprilTagFamilyId

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    from PIL import Image
except ImportError as e:
    e.add_note('to use PDF generation features, install this package with the "generate" extra')
    raise

from ..apriltag.render.image import AprilTagImageGenerator

__all__ = [
    'generate_apriltag_grid_pdf',
]

MM_OVER_PIXELS = 127 / 480


def generate_apriltag_grid_pdf(first_tag_id: int,
                               tag_size_mm: float,
                               grid: tuple[int, int],
                               tag_family: AprilTagFamilyId = 'tagStandard41h12',
                               outfile: Path = Path('tags.pdf'),
                               page_format: Literal['a3', 'a4', 'a5', 'letter', 'legal'] | tuple[
                                   int | float, int | float] = 'letter',
                               orientation: Literal['portrait', 'landscape'] = 'portrait',
                               font_size: int = 12,
                               margin_width_mm: float = 6.0,
                               tag_padding_mm: float = 5.0,
                               tag_horizontal_spacing_mm: float = 15.0,
                               tag_vertical_spacing_mm: float = 15.0) -> None:
    """
    Generates a PDF file with the given AprilTags.
    :param tag_ids: A sequence of AprilTag IDs. Currently only a limited number of IDs are supported.
    :param tag_size_mm: The distance between the tag's corner points in millimeters. Be sure to verify how the corner
       points are defined for the family you are using (for the default family, the corner points are the corners of
       the interior square).
    :param tag_family: The AprilTag family for which tags will be generated (default: ``tagStandard41h12``).
    :param page_format: The size of the PDF. This may be specified as a string for certain common defaults
       or as a tuple indicating the size of the horizontal and vertical axes in millimeters (default: ``"letter"``).
    :param outfile: The path to which the resulting PDF will be written (default: ``./tags.pdf``).
    """
    if tag_horizontal_spacing_mm < tag_padding_mm * 2:
        raise ValueError('tag horizontal spacing must be greater than twice the tag padding')
    if tag_vertical_spacing_mm < tag_padding_mm * 2:
        raise ValueError('tag vertical spacing must be greater than twice the tag padding')
    apriltag_image_generator = AprilTagImageGenerator(family_name=tag_family)

    tag_image_size_mm = (tag_size_mm
                         * apriltag_image_generator.family.total_width
                         / apriltag_image_generator.family.width_at_border)

    pdf = FPDF(orientation=orientation, unit='mm', format=page_format)
    pdf.set_font(family='Helvetica', size=font_size)
    pdf.set_auto_page_break(False)
    pdf.add_page(same=True)
    pdf.set_margin(margin_width_mm)

    if grid[1] > 1:
        grid_width_mm = grid[1] * tag_image_size_mm + (grid[1] - 1) * tag_horizontal_spacing_mm
    else:
        grid_width_mm = tag_image_size_mm
    if grid_width_mm > pdf.w - pdf.l_margin - pdf.r_margin:
        raise ValueError('grid is too large to fit on the page in the horizontal axis')
    grid_left_x_mm = (pdf.w - grid_width_mm) / 2
    grid_right_x_mm = (pdf.w + grid_width_mm) / 2

    if grid[0] > 1:
        grid_height_mm = grid[0] * tag_image_size_mm + (grid[0] - 1) * tag_vertical_spacing_mm
    else:
        grid_height_mm = tag_image_size_mm
    if grid_height_mm > pdf.h - pdf.t_margin - pdf.b_margin:
        raise ValueError('grid is too large to fit on the page in the vertical axis')
    grid_top_y_mm = (pdf.h - grid_height_mm) / 2
    grid_bottom_y_mm = (pdf.h + grid_height_mm) / 2

    tag_left_x = grid_left_x_mm + (tag_image_size_mm + tag_horizontal_spacing_mm) * np.arange(grid[1])
    tag_top_y = grid_top_y_mm + (tag_image_size_mm + tag_vertical_spacing_mm) * np.arange(grid[0])

    for tag_id, (x, y) in zip(count(first_tag_id), product(range(grid[1]), range(grid[0]))):
        tag_image = apriltag_image_generator.generate_image(tag_id=tag_id,
                                                            tag_size=2 * round(tag_image_size_mm / MM_OVER_PIXELS))
        pdf.image(tag_image,
                  x=float(tag_left_x[x]),
                  y=float(tag_top_y[y]),
                  w=tag_image_size_mm,
                  h=tag_image_size_mm,
                  keep_aspect_ratio=True)

    tag_right_x = tag_left_x[:-1] + tag_image_size_mm
    for y in (tag_image_size_mm / 2) + tag_top_y:
        pdf.line(x1=pdf.l_margin,
                 y1=y,
                 x2=grid_left_x_mm - tag_padding_mm,
                 y2=y)
        for x in map(float, tag_right_x):
            pdf.line(x1=x + tag_padding_mm,
                     y1=y,
                     x2=x + tag_horizontal_spacing_mm - tag_padding_mm,
                     y2=y)
        pdf.line(x1=grid_right_x_mm + tag_padding_mm,
                 y1=y,
                 x2=pdf.w - pdf.r_margin,
                 y2=y)

    tag_bottom_y = tag_top_y[:-1] + tag_image_size_mm
    for x in (tag_image_size_mm / 2) + tag_left_x:
        pdf.line(x1=x,
                 y1=pdf.t_margin,
                 x2=x,
                 y2=grid_top_y_mm - tag_padding_mm)
        for y in map(float, tag_bottom_y):
            pdf.line(x1=x,
                     y1=y + tag_padding_mm,
                     x2=x,
                     y2=y + tag_vertical_spacing_mm - tag_padding_mm)
        pdf.line(x1=x,
                 y1=grid_bottom_y_mm + tag_padding_mm,
                 x2=x,
                 y2=pdf.h - pdf.b_margin)

    pdf.set_xy(pdf.l_margin, pdf.t_margin)
    pdf.cell(text=f'IDs {first_tag_id} to {first_tag_id + (grid[0] * grid[1]) - 1}',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(text=f'Size: {tag_size_mm:.4} mm',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    tag_horizontal_separation_mm = tag_image_size_mm + tag_horizontal_spacing_mm
    tag_vertical_separation_mm = tag_image_size_mm + tag_vertical_spacing_mm
    pdf.cell(text=f'Separation: ({tag_horizontal_separation_mm:.4} mm, {tag_vertical_separation_mm:.4} mm)')

    pdf.set_title('AprilTag grid')
    pdf.set_creator('apriltag_pose_estimation.generate')
    pdf.output(str(outfile))
