from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from .._criterion import Criterion


class ImageCriterion(Criterion):
    """A criterion that only considers images for evaluation."""

    def __init__(self, inverse: bool = False, allow_batched=False) -> None:
        """
        Initialize the Image Criterion.

        :param inverse: If true, invert the measure.
        :param allow_batched: If true, allow batching.
        """
        super().__init__(inverse, allow_batched)

        if self._allow_batched:
            eval_func = self.evaluate

            def batched_evaluate(
                self: Any, *, images: list[Tensor], batch_dim: Optional[int] = None, **kwargs: Any
            ) -> Union[float, list[float]]:
                """
                A wrapper to the classifier criterion evaluate function.

                :param self: Self-explanatory :D.
                :param images: The images to evaluate the criterion on.
                :param batch_dim: The batch dimension.
                :param kwargs: The KW-Args parsed.
                :returns: The value(s).
                """
                if batch_dim is None:
                    images = [i.unsqueeze(0) for i in images]
                elif batch_dim != 0:
                    images = [i.transpose(0, batch_dim) for i in images]
                results = eval_func(images=images, **kwargs)
                return results[0] if batch_dim is None and isinstance(results, list) else results

            self.evaluate = batched_evaluate.__get__(self, self.__class__)  # type: ignore[method-assign]

    @abstractmethod
    def evaluate(self, *, images: list[Tensor], **kwargs: Any) -> Union[float, list[float]]:
        """
        Evaluate the criterion in question.

        :param images: The images to evaluate the criterion on.
        :param kwargs: The KW-Args parsed.
        :returns: The value(s).
        """
        ...

    @staticmethod
    def _prepare_tensor(tensor: Union[Tensor, NDArray], batch_dim: Optional[int] = None) -> NDArray:
        """
        Prepare torch Tensor into numpy NDArray with correct dimension order.

        :param tensor: The tensor to prepare.
        :param batch_dim: The batch dimension if existing.
        :returns: The numpy array.
        """
        if isinstance(tensor, np.ndarray):
            return tensor

        if batch_dim is None:
            tensor = tensor.unsqueeze(0)  # Unsqueeze single dimension.
        elif batch_dim != 0:
            tensor = tensor.transpose(0, batch_dim)
        tensor.permute(0, 2, 3, 1)
        ndarray = tensor.detach().cpu().numpy()
        return ndarray[0] if batch_dim is None else ndarray

    def prepare_images(
        self, images: list[Tensor], batch_dim: Optional[int] = None
    ) -> tuple[NDArray, NDArray]:
        """
        Prepare image pairs for evaluation and assert that there are two images.

        :param images: The images to prepare.
        :param batch_dim: The batch dimension.
        :returns: The prepared images as a tuple.
        """
        assert len(images) == 2, f"ERROR, {self._name} requires 2 images, found {len(images)}"
        images = [self._prepare_tensor(i, batch_dim=batch_dim) for i in images]
        return images[0], images[1]
