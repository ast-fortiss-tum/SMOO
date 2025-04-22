from typing import Any

from sewar import uqi
from torch import Tensor

from ._image_criterion import ImageCriterion


class UQI(ImageCriterion):
    """Implements the universal image quality index using sewar."""

    _name: str = "UQI"

    def evaluate(self, *, images: list[Tensor], **_: Any) -> float:
        """
        Get the Universal Image Quality Index score.

        This score is in range (0,1) with 0 being the optimum.

        :param images: Images to compare.
        :param _: Additional unused kwargs.
        :returns: SSIM score.
        """
        i1, i2 = self.prepare_images(images)
        return 1 - uqi(i1, i2) if self._inverse else uqi(i1, i2)
