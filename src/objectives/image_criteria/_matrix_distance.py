from typing import Any, Union

import torch
from torch import Tensor

from ._image_criterion import ImageCriterion


class MatrixDistance(ImageCriterion):
    """Implements a channel-wise matrix distance measure based on torch.linalg.norm."""

    _name: str = "MatrixDistance"
    _all_norms: list[str] = ["fro", "nuc", "inf", "-inf", "1", "-1", "2", "-2"]

    def __init__(self, inverse: bool = False, norm: str = "fro") -> None:
        """
        Initialize the MatrixDistance criterion.

        :param inverse: Whether the measure should be inverted (default: False).
        :param norm: Which norm to use (default: fro).
        """
        super().__init__(inverse)
        assert norm in self._all_norms, f"Norm {norm} not in supported norms: {self._all_norms}"
        self.norm = norm
        self._name += f"_{norm}"

    @torch.no_grad()
    def evaluate(
        self, *, images: list[Tensor], batch_dim: Union[int, None], **_: Any
    ) -> Union[float, list[float]]:
        """
        Calculate the normalized matrix distance between two tensors that range [0,1].

        :param images: Images to compare.
        :param batch_dim: The batch dimension.
        :param _: Additional unused kwargs.
        :returns: The distance.
        """
        i1, i2 = images  # Expect the image tensors to have shape: B x C x H x W
        if batch_dim is None:
            i1 = i1.unsqueeze(0)
            i2 = i2.unsqueeze(0)
        elif batch_dim != 1:
            i1 = i1.transpose(0, batch_dim)
            i2 = i2.transpose(0, batch_dim)
        # Upper bound of distance.
        ub = torch.linalg.matrix_norm(torch.ones_like(i1), self.norm, dim=(-2, -1))

        diff = i1 - i2
        frob = torch.linalg.matrix_norm(diff, self.norm, dim=(-2, -1))
        scaled = frob / ub

        channel_wise = scaled.mean(dim=1)
        inverted = torch.abs(self._inverse.real - channel_wise).tolist()
        return inverted[0] if batch_dim is None else inverted
