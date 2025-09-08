from typing import Any

from torch import Tensor
from torch.nn.modules.loss import _Loss

from ._classifier_criterion import ClassifierCriterion


class TorchLossCriterion(ClassifierCriterion):
    """A wrapper to torch loss functions."""

    _name: str = "TorchLoss"

    def __init__(self, loss_fn: _Loss) -> None:
        """
        Initialize the Criterion.

        :param loss_fn: The loss function to use.
        """
        super().__init__(inverse=False, allow_batched=True)
        self._name += str(loss_fn.__class__.__name__)
        self._loss_fn = loss_fn

    def evaluate(self, *, logits: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
        """
        Calculate the loss.

        :param logits: Logits tensor.
        :param target: Target tensor.
        :param kwargs: Other Kwargs to use.
        :returns: The value.
        """
        return self._loss_fn(logits, target, **kwargs)
