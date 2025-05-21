from typing import Any

import numpy as np
from torch import Tensor

from ._image_criterion import ImageCriterion


class CosDissimilarity(ImageCriterion):
    """Implements cos dissimilarity measure."""

    _name: str = "CosDissim"

    def evaluate(self, *, images: list[Tensor], **_: Any) -> float:
        """
        Get the cosine-dissimilarity between two images.

        Range [0,1] with 0 being the same image.

        :param images: Images to compare.
        :param _: Additional unused kwargs.
        :returns: The score.
        """
        i1, i2 = self.prepare_images(images)

        value = np.dot(i1.flatten(), i2.flatten()) / (np.linalg.norm(i1) * np.linalg.norm(i2))
        return 1 - value
