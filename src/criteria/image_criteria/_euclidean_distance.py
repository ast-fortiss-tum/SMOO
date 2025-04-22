from typing import Any

import numpy as np
from torch import Tensor

from ._image_criterion import ImageCriterion


class EuclideanDistance(ImageCriterion):
    """Implements a Euclidean Distance measure."""

    _name: str = "EuclideanDistance"
    _normalize: bool

    def __init__(self, normalize: bool = False) -> None:
        """
        Initialize the EuclideanDistance measure.

        :param normalize: Whether to normalize the computed distance [0,1].
        """
        super().__init__()
        self._normalize = normalize

    def evaluate(self, *, images: list[Tensor], **_: Any) -> float:
        """
        Calculate the normalized frobenius distance between two tensors that range [0,1].

        :param images: Images to compare.
        :param _: Additional unused kwargs.
        :returns: The distance.
        """
        i1, i2 = self.prepare_images(images)
        return (
            np.linalg.norm(i1 - i2)
            if not self._normalize
            else np.linalg.norm(i1 - i2) / np.sqrt(len(i1))
        )
