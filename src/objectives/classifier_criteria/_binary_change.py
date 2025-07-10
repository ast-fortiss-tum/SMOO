from typing import Any, Optional

from torch import Tensor

from ._classifier_criterion import ClassifierCriterion


class BinaryChange(ClassifierCriterion):
    """Implements a criterion that observes change in binary classification."""

    _name: str = "BinaryChange"
    _precondition_logits: Optional[Tensor]

    def __init__(
        self,
        target_logit: Optional[int] = None,
        flip_sign: bool = False,
        inverse: bool = False,
        v_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        Initialize the BinaryChange Criterion.

        By default, the change is expected to be a decrease in confidence.

        :param target_logit: Which logit should be observed for change (some classifiers have multiple binary outputs).
        :param flip_sign: Whether the sign should be flipped instead of pure increase / decrease of confidence.
        :param inverse: Whether the criterion should be inverted.
        :param v_range: Set a range of the input if applicable, to normalize outputs to [0,1].
        :raises NotImplementedError: If target_logit is None.
        """
        super().__init__(inverse=inverse, allow_batched=True)
        self._target_logit = target_logit
        self._v_range = v_range or (0.0, 1.0)
        self._flip_sign = flip_sign

        if target_logit is None:
            raise NotImplementedError(
                "Target logit must be specified. Or implement handling for returning multiple :)."
            )

    def evaluate(self, *, logits: Tensor, **_: Any) -> list[float]:
        """
        Calculate the change in binary confidence values.

        This function returns normalized values if the v_range was set in initialization.

        :param logits: Logits tensor.
        :param _: Unused kwargs.
        :returns: The value of the change.
        :raises ValueError: If precondition logits are not set before flipping sign.
        """
        logits = logits[:, self._target_logit] if self._target_logit else logits

        if self._flip_sign:
            if self._precondition_logits is None:
                raise ValueError("Precondition logits must be set before flipping sign.")
            partial = logits * self._precondition_logits
        else:
            partial = (-1) ** (2 - self._inverse.real) * logits
        partial = (partial - self._v_range[0]) / (self._v_range[1] - self._v_range[0])
        results = partial.tolist()
        return results

    def precondition(self, *, logits: Tensor, **_: Any) -> None:
        """
        Calculate the change in binary confidence values.

        This function returns normalized values if the v_range was set in initialization.

        :param logits: Logits tensor.
        :param _: Unused kwargs.
        """
        self._precondition_logits = logits[:, self._target_logit] if self._target_logit else logits
