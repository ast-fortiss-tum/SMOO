from typing import Any

from sewar import msssim
from torch import Tensor

from ._image_criterion import ImageCriterion


class MSSSIM(ImageCriterion):
    """Implements the Multi-Scale SSIM using sewar."""

    _name: str = "MS-SSIM"

    def __init__(self) -> None:
        """Initialize the Multi-Scale SSIM criterion."""
        super().__init__()

    def evaluate(self, *, images: list[Tensor], **_: Any) -> float:
        """
        Get the Multi-Scale SSIM score.

        This score is in range (0,1) with 0 being the optimum.

        :param images: The images to compare.
        :param _: Additional unused kwargs.
        :returns: The score.
        """
        i1, i2 = self.prepare_images(images)
        i1 = i1.transpose(1, 2, 0)
        i2 = i2.transpose(1, 2, 0)
        return 1 - msssim(i1, i2, MAX=1.0).real if self._inverse else msssim(i1, i2, MAX=1.0).real
