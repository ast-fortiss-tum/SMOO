from abc import abstractmethod
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from .._criterion import Criterion


class ImageCriterion(Criterion):
    """A criterion that only considers images for evaluation."""

    @abstractmethod
    def evaluate(self, *, images: list[Tensor], **kwargs: Any) -> float:
        """
        Evaluate the criterion in question.

        :param images: The images to evaluate the criterion on.
        :param kwargs: The KW-Args parsed.
        :returns: The value(s).
        """
        ...

    @staticmethod
    def _prepare_tensor(tensor: Union[Tensor, NDArray]) -> NDArray:
        """
        Prepare torch Tensor into numpy NDArray with correct dimension order.

        :param tensor: The tensor to prepare.
        :returns: The numpy array.
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        ndarray = tensor.detach().cpu().numpy()
        ndarray = ndarray.transpose(1, 2, 0)
        return ndarray

    def prepare_images(self, images: list[Tensor]) -> tuple[NDArray, NDArray]:
        """
        Prepare image pairs for evaluation and assert that there are two images.

        :param images: The images to prepare.
        :returns: The images prepared in form of a tuple.
        """
        assert len(images) == 2, f"ERROR, {self._name} requires 2 images, found {len(images)}"
        images = [self._prepare_tensor(i) for i in images]
        return images[0], images[1]
