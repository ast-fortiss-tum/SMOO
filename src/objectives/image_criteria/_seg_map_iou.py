from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor
from torchvision.transforms.functional import gaussian_blur

from ._image_criterion import ImageCriterion


class SegMapIoU(ImageCriterion):
    """Computes IoU of segmentation maps."""

    _name: str = "SegMapIoU"
    _blur: Callable[[Tensor], Tensor]

    def __init__(self, gaussian_params: Optional[tuple[int, float]], inverse: bool = False) -> None:
        """
        Initialize the IoU measure for segmentation maps.
        Note that segmentation maps are expected to be H x W matrices with integer values from 0-inf.
        By default, this measure computes 1-IoU as we want to minimize in optimization, this can be changed by setting inverse to True.

        :param gaussian_params: Parameters for gaussian blurring (kernel_size:int, sigma:float). If None, no blurring is applied.
        :param inverse: Whether the measure should be inverted (default: False).
        """
        super().__init__(inverse)
        self._blur = (
            (lambda x: x)
            if gaussian_params is None
            else (lambda x: gaussian_blur(x, *gaussian_params))
        )

    @torch.no_grad()
    def evaluate(
        self, *, images: list[Tensor], batch_dim: Optional[int], **_: Any
    ) -> Union[float, list[float]]:
        """
        Calculate the Segmentation Map IoU of two images.

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
        # TODO: get seg maps from images
        seg_i1, seg_i2 = ..., ...

        bound_i1, bound_i2 = self._get_boundary_map(seg_i1), self._get_boundary_map(seg_i2)
        final_i1, final_i2 = self._blur(bound_i1), self._blur(bound_i2)
        final_sum = final_i1 + final_i2
        c1, c2 = (final_sum > final_i1), (final_sum > final_i2)
        iou = torch.logical_and(c1, c2).sum() / torch.logical_or(c1, c2).sum()

        inverted_tensor: Tensor = iou if self._inverse else 1 - iou
        inverted: list[float] = inverted_tensor.float().tolist()
        return inverted[0] if batch_dim is None else inverted

    @staticmethod
    def _get_boundary_map(seg: Tensor) -> Tensor:
        """
        Get boundaries of segmentation regions.

        :param seg: The Segmentation map.
        :return: The boundary map where all elements are in {0,1}, 1 being an edge 0 nothing.
        """
        edge_y = seg[:, 1:, :] != seg[:, :-1, :]
        edge_x = seg[:, :, 1:] != seg[:, :, :-1]

        edge_map = torch.zeros_like(seg, dtype=torch.float32)
        edge_map[:, 1:, :] += edge_y
        edge_map[:, :, 1:] += edge_x
        return (edge_map > 0).type(torch.float32)
