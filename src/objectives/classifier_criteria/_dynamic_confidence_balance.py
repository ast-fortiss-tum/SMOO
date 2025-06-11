from typing import Any, Optional, Union

import torch
from torch import Tensor

from ._classifier_criterion import ClassifierCriterion


class DynamicConfidenceBalance(ClassifierCriterion):
    """Implements a dynamic confidence balance measure."""

    _name: str = "DynCB"
    _target_primary: bool

    def __init__(self, inverse: bool = False, target_primary: Optional[bool] = None) -> None:
        """
        Initialize the criterion.

        :param inverse: Whether the measure should be inverted.
        :param target_primary: Whether y1 is focus of the measure or yp, if none neither is in focus.
        """
        super().__init__(inverse=inverse)
        self._target_primary = target_primary

    def evaluate(
        self, *, logits: Tensor, label_targets: list[int], batch_dim: Optional[int] = None, **_: Any
    ) -> Union[float, list[float]]:
        """
        Calculate the confidence balance of 2 confidence values.

        This functions assumes input range of [0, 1].

        :param logits: Logits tensor.
        :param label_targets: Label targets used to determine targets of balance.
        :param batch_dim: Which dim to use for batchwise operations.
        :param _: Unused kwargs.
        :returns: The value.
        """
        origin = label_targets[0]  # The primary class
        if batch_dim is None:
            logits = logits.unsqueeze(0)
        elif batch_dim != 0:
            logits = logits.transpose(0, batch_dim)

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, origin] = True
        second_term = logits.masked_fill(mask, float("-inf")).max(dim=1).values

        s = logits[:, origin] + second_term
        d = torch.abs(logits[:, origin] - second_term)

        target = 0
        if self._target_primary is True:
            target = second_term
        elif self._target_primary is False:
            target = logits[:, origin]

        result = torch.abs(self._inverse.real - target - d / s).tolist()
        return result[0] if batch_dim is None else result
