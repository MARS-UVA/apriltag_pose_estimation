"""Provides a function for generating PDF documents with AprilTags for printing with a traditional printer."""
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

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
    'generate_apriltag_pdf',
]

MM_OVER_PIXELS = 127 / 480


def generate_apriltag_pdf(tag_ids: Iterable[int],
                          tag_size_mm: float,
                          tag_family: AprilTagFamilyId = 'tagStandard41h12',
                          outfile: Path = Path('tags.pdf'),
                          page_format: Literal['a3', 'a4', 'a5', 'letter', 'legal'] | tuple[
                              int | float, int | float] = 'letter',
                          orientation: Literal['portrait', 'landscape'] = 'portrait',
                          font_size: int = 32,
                          margin_width_mm: float = 6.0,
                          tag_padding_mm: float = 5.0) -> None:
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
    apriltag_image_generator = AprilTagImageGenerator(family_name=tag_family)

    tag_image_size_mm = (tag_size_mm
                         * apriltag_image_generator.family.total_width
                         / apriltag_image_generator.family.width_at_border)

    pdf = FPDF(orientation=orientation, unit='mm', format=page_format)
    pdf.set_font(family='Helvetica', size=font_size)
    pdf.set_auto_page_break(False)

    center_x = pdf.w / 2
    center_y = pdf.h / 2
    tag_image_left_x = (pdf.w - tag_image_size_mm) / 2
    tag_image_top_y = (pdf.h - tag_image_size_mm) / 2
    tag_image_right_x = (pdf.w + tag_image_size_mm) / 2
    tag_image_bottom_y = (pdf.h + tag_image_size_mm) / 2

    for tag_id in tag_ids:
        pdf.add_page(same=True)
        pdf.set_margin(margin_width_mm)
        tag_image = apriltag_image_generator.generate_image(tag_id=tag_id,
                                                            tag_size=2 * round(tag_image_size_mm / MM_OVER_PIXELS))
        pdf.image(tag_image,
                  x=tag_image_left_x,
                  y=tag_image_top_y,
                  w=tag_image_size_mm,
                  h=tag_image_size_mm,
                  keep_aspect_ratio=True)

        pdf.line(x1=pdf.l_margin,
                 y1=center_y,
                 x2=tag_image_left_x - tag_padding_mm,
                 y2=center_y)
        pdf.line(x1=tag_image_right_x + tag_padding_mm,
                 y1=center_y,
                 x2=pdf.w - pdf.r_margin,
                 y2=center_y)

        pdf.line(x1=center_x,
                 y1=pdf.t_margin,
                 x2=center_x,
                 y2=tag_image_top_y - tag_padding_mm)
        pdf.line(x1=center_x,
                 y1=tag_image_bottom_y + tag_padding_mm,
                 x2=center_x,
                 y2=pdf.h - pdf.b_margin)

        pdf.set_xy(pdf.l_margin, pdf.t_margin)
        pdf.cell(text=f'ID {tag_id}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(text=f'Size: {tag_size_mm:.4} mm')

    pdf.set_title('Set of AprilTags')
    pdf.set_creator('apriltag_pose_estimation.generate')
    pdf.output(str(outfile))
