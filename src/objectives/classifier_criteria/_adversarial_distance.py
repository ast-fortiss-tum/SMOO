from ._classifier_criterion import ClassifierCriterion
from torch import Tensor
import torch
from typing import Any

class AdversarialDistance(ClassifierCriterion):
    """A objective function that allows to optimize for adversarial examples."""
    _name: str = "AdvD"

    def __init__(self, target_pair: bool = False, inverse: bool=False) -> None:
        """
        Initialize the AdversarialDistance objective.

        :param target_pair: If true, we want to target the adversarial examples of specific class pairs (class X -> class Y) else (class X -> any class).
        :param inverse: Whether the measure should be inverted.
        """
        super().__init__(inverse=inverse)
        self._target_pair = target_pair

    def evaluate(self, *, logits: Tensor, label_targets: list[int], **_: Any) -> float:
        """
        Calculate the confidence balance of 2 confidence values.

        This functions assumes input range of [0, 1].

        :param logits: Logits tensor.
        :param label_targets: Label targets used to determine targets of balance.
        :param _: Unused kwargs.
        :returns: The value.
        """
        origin, target, *_ = label_targets
        mask = torch.arange(logits.size(0), device=logits.device) == origin
        second_term = (logits[target].item() if self._target_pair else logits[~mask].max())
        return  second_term-logits[origin].item() if self._inverse.real else logits[origin].item()-second_term
