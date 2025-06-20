from typing import Any, Optional

from torch import Tensor

from .._criterion import Criterion


class Sum(Criterion):
    """Implements a sum-based criterion."""

    _min_max: Optional[tuple[float, float]] = None
    _name: str = "Sum"

    def __init__(self, min_max: Optional[tuple[float, float]] = None) -> None:
        """
        Initialize the Sum criterion.

        :param min_max: The min and max values for the sum to use for normalization.
        """
        super().__init__()
        self._min_max = min_max or (0, 1)

    def evaluate(
        self,
        *,
        element: Tensor,
        **_: Any,
    ) -> float:
        """
        Get the sum of the element.

        :param element: Element to sum.
        :param _: Additional unused args.
        :return: The sum.
        """
        minv, maxv = self._min_max
        result = (element.sum() - minv) / (maxv - minv)
        return result
