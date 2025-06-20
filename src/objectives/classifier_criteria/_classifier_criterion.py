from abc import abstractmethod
from typing import Any, Union

from torch import Tensor

from .._criterion import Criterion


class ClassifierCriterion(Criterion):
    """A criterion that only considers classifier outputs."""

    def __init__(self, inverse: bool = False, allow_batched: bool = False) -> None:
        """
        Initialize the ClassifierCriterion.

        :param inverse: Whether the criterion should be inverted.
        :param allow_batched: Whether the criterion supports batching.
        """
        super().__init__(inverse, allow_batched)

        if self._allow_batched:
            eval_func = self.evaluate

            def batched_evaluate(
                _,
                *,
                logits: Tensor,
                label_targets: list[int],
                batch_dim: Union[int, None] = None,
                **kwargs: Any
            ) -> Union[float, list[float]]:
                """
                A wrapper to the classifier criterion evaluate function.

                :param _: This will be self.
                :param logits: The logits tensor produced by the model.
                :param label_targets: The actual labels of the predicted elements.
                :param batch_dim: The dimension used for batching (default: None).
                :param kwargs: Other keyword arguments passed to the criterion.
                :returns: The value(s).
                """
                if batch_dim is None:
                    logits = logits.unsqueeze(0)
                elif batch_dim != 0:
                    logits = logits.transpose(0, batch_dim)
                results = eval_func(logits=logits, label_targets=label_targets, **kwargs)
                return results[0] if batch_dim is None else results

            self.evaluate = batched_evaluate.__get__(self, self.__class__)

    @abstractmethod
    def evaluate(
        self, *, logits: Tensor, label_targets: list[int], **kwargs: Any
    ) -> Union[float, list[float]]:
        """
        Evaluate the criterion in question.

        :param logits: The logits tensor produced by the model.
        :param label_targets: The actual labels of the predicted elements.
        :param kwargs: Other keyword arguments passed to the criterion.
        :returns: The value(s).
        """
        ...
