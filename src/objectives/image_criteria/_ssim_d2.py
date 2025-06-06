from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from torch import Tensor

from ._image_criterion import ImageCriterion


class SSIMD2(ImageCriterion):
    """
    Implements SSIM metric.

    Implementation based on https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/metrics/_structural_similarity.py#L15-L292.
    And https://ece.uwaterloo.ca/~z70wang/publications/TIP_SSIM_MathProperties.pdf.
    Due to numpy conflicts with cuda we had to do our own implementation.
    """

    """Parameters for the evaluation."""
    truncate: float
    sigma: float
    k1: float
    k2: float

    def __init__(
        self, truncate: float = 3.5, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03
    ) -> None:
        """
        Initialize the SSIM D2 metric.

        :param truncate: Truncation value for the gaussian.
        :param sigma: Sigma value for the gaussian.
        :param k1: The K1 coefficient.
        :param k2: The K2 coefficient.
        """
        self.truncate, self.sigma, self.k1, self.k2 = truncate, sigma, k1, k2
        self._name = "SSIM-D2"
        super().__init__()

    def evaluate(
        self,
        *,
        images: list[Tensor],
        **_: Any,
    ) -> float:
        """
        Get structural similarity between two images as D_2 metric.

        :param images: Images to compare.
        :param _: Additional unused kwargs.
        :returns: SSIM score.
        """
        i1, i2 = self.prepare_images(images)

        assert (
            i1.shape == i2.shape
        ), f"Error: Both images need to be of same size ({i1.shape}, {i2.shape})."

        score = self._ssim_d2(i1, i2)
        return score

    def _ssim_d2(self, i1: NDArray, i2: NDArray) -> float:
        """
        Compute the SSIM D2 metric https://ece.uwaterloo.ca/~z70wang/publications/TIP_SSIM_MathProperties.pdf.

        :param i1: First image.
        :param i2: Second image.
        :returns: SSIM score.
        """
        filter_curry = lambda image: gaussian_filter(
            image, sigma=self.sigma, truncate=self.truncate
        )
        pad = (2 * int(self.truncate * self.sigma + 0.5)) // 2

        ux, uy = filter_curry(i1), filter_curry(i2)  # local mean of x and y
        uxx, uyy, uxy = filter_curry(i1 * i1), filter_curry(i2 * i2), filter_curry(i1 * i2)

        vx = uxx - ux * ux  # local variance of x
        vy = uyy - uy * uy  # local variance of y
        vxy = uxy - ux * uy  # local covariance between x and y

        c1 = (self.k1 * 1) ** 2.0  # (K1 * Data-Range)²
        c2 = (self.k2 * 1) ** 2.0  # (K2 * Data-Range)²

        a1 = 2.0 * ux * uy + c1
        a2 = 2.0 * vxy + c2
        b1 = ux**2.0 + uy**2.0 + c1
        b2 = vx + vy + c2

        s1 = np.clip(a1 / b1, 0, 1)
        s2 = np.clip(a2 / b2, 0, 1)
        d = np.sqrt(2.0 - s1 - s2)

        d2 = d[pad:-pad, pad:-pad, :].mean()
        return d2 / np.sqrt(2)
