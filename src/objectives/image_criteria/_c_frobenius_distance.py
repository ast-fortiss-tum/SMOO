from typing import Any, Union

import torch
from torch import Tensor

from ._image_criterion import ImageCriterion


class CFrobeniusDistance(ImageCriterion):
    """Implements a channel-wise Frobenius Distance measure."""

    _name: str = "CFrobDistance"

    @torch.no_grad()
    def evaluate(
        self, *, images: list[Tensor], batch_dim: Union[int, None], **_: Any
    ) -> Union[float, list[float]]:
        """
        Calculate the normalized frobenius distance between two tensors that range [0,1].

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
        ub = torch.linalg.norm(
            torch.ones_like(i1), "fro", dim=(-2, -1)
        )  # Upper bound of frobenius distance case of image [0,1].

        diff = i1 - i2
        frob = torch.linalg.norm(diff, "fro", dim=(-2, -1))
        scaled = frob / ub

        channel_wise = scaled.mean(dim=1)
        inverted = torch.abs(self._inverse.real - channel_wise).tolist()
        return inverted[0] if batch_dim is None else inverted
