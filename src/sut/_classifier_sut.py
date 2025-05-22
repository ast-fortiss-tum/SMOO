from torch import  Tensor, nn

from ._sut import SUT
from .auxiliary_components import MonteCarloDropoutScaffold


class ClassifierSUT(SUT):
    """A classifier SUT."""

    _model: nn.Module
    _softmax: nn.Softmax

    _apply_softmax: bool

    def __init__(self, model: nn.Module, apply_softmax: bool = False, use_mcd: bool = False) -> None:
        """
        Initialize the classifier SUT.

        :param model: The model to use.
        :param apply_softmax: Whether to apply softmax or not.
        :param use_mcd: Whether to use Monte Carlo Dropout or not.
        """
        self._model = MonteCarloDropoutScaffold(model) if use_mcd else model
        self._softmax = nn.Softmax(dim=1)

        self._apply_softmax = apply_softmax

    def process_input(self, inpt: Tensor) -> Tensor:
        """
        Predict class probabilities from input.

        :param inpt: Input tensor.
        :return: Predicted class probabilities.
        """
        result = self._model(inpt)
        return self._softmax(result) if self._apply_softmax else result
