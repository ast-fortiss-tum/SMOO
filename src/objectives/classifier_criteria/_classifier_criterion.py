from abc import abstractmethod
from typing import Any

from torch import Tensor

from .._criterion import Criterion


class ClassifierCriterion(Criterion):
    """A criterion that only considers classifier outputs."""

    @abstractmethod
    def evaluate(self, *, logits: Tensor, label_targets: list[int], **kwargs: Any) -> float:
        """
        Evaluate the criterion in question.

        :param logits: The logits tensor produced by the model.
        :param label_targets: The actual labels of the predicted elements.
        :param kwargs: Other keyword arguments passed to the criterion.
        :returns: The value(s).
        """
        ...
