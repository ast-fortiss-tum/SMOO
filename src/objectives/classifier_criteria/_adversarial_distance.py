from typing import Any, Union

import torch
from torch import Tensor

from ._classifier_criterion import ClassifierCriterion


class AdversarialDistance(ClassifierCriterion):
    """An objective function that allows to optimize for adversarial examples."""

    _name: str = "AdvD"

    def __init__(
        self, target_pair: bool = False, inverse: bool = False, exp_decay_lambda: float = None
    ) -> None:
        """
        Initialize the AdversarialDistance objective.

        :param target_pair: If true, we want to target the adversarial examples of specific class pairs (class X -> class Y) else (class X -> any class).
        :param inverse: Whether the measure should be inverted.
        :param exp_decay_lambda: Factor for exponential decay f(x) = e^(-lambda*x).
        :raises NotImplementedError: If inverse of funcion is required.
        """
        super().__init__(inverse=inverse)
        self._target_pair = target_pair
        self._exp_decay_lambda = exp_decay_lambda
        if self._inverse:
            raise NotImplementedError("Inverse does not function properly yet.")

    def evaluate(
        self,
        *,
        logits: Tensor,
        label_targets: list[int],
        batch_dim: Union[int, None] = None,
        **_: Any
    ) -> Union[float, list[float]]:
        """
        Calculate the confidence balance of 2 confidence values.

        This function assumes an input range of [0, 1].

        :param logits: Logits tensor.
        :param label_targets: Label targets used to determine targets of balance.
        :param batch_dim: Batch dimension if evaluation is done batch wise.
        :param _: Unused kwargs.
        :returns: The value in range [0,1].
        """
        origin, target, *_ = label_targets

        if batch_dim is None:
            logits = logits.unsqueeze(0)
        elif batch_dim != 0:
            logits = logits.transpose(0, batch_dim)

        if self._target_pair:
            second_term = logits[target].item()
        else:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, origin] = True
            second_term = logits.masked_fill(mask, float("-inf")).max(dim=1).values

        partial = (-1) ** (2 - self._inverse.real) * (
            logits[:, origin] - second_term
        ) + self._inverse.real
        partial = (partial + 1) / 2
        if self._exp_decay_lambda is not None:
            partial = torch.exp(-self._exp_decay_lambda * partial)
        results = partial.tolist()
        return partial[0] if batch_dim is None else results
