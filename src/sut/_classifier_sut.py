from typing import Union

import torch
from torch import Tensor, nn, no_grad

from ._sut import SUT
from .auxiliary_components import MonteCarloDropoutScaffold


class ClassifierSUT(SUT):
    """A classifier SUT."""

    _model: nn.Module
    _softmax: nn.Softmax

    _apply_softmax: bool
    _batch_size: int

    def __init__(
        self,
        model: nn.Module,
        apply_softmax: bool = False,
        use_mcd: bool = False,
        batch_size: int = 0,
        device: Union[torch.device, None] = None,
    ) -> None:
        """
        Initialize the classifier SUT.

        :param model: The model to use.
        :param apply_softmax: Whether to apply softmax or not.
        :param use_mcd: Whether to use Monte Carlo Dropout or not.
        :param batch_size: The batch size to use for prediction.
        :param device: The device to use if available.
        """
        self._model = MonteCarloDropoutScaffold(model) if use_mcd else model
        self._model.eval()
        self._softmax = nn.Softmax(dim=-1)

        self._apply_softmax = apply_softmax
        self._batch_size = batch_size
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_device(self._device)

    def set_device(self, device: torch.device) -> None:
        """
        Set the device which the model is on.

        :param device: The device to use.
        """
        self._model.to(device)

    def process_input(self, inpt: Tensor) -> Tensor:
        """
        Predict class probabilities from input.

        :param inpt: Input tensor.
        :return: Predicted class probabilities on CPU.
        """
        if inpt.device != self._device:
            inpt = inpt.to(self._device)

        batch_size = max(
            self._batch_size or inpt.size(0), 1
        )  # If batchsize == 0 -> do whole input.
        n_chunks = (inpt.size(0) + batch_size - 1) // batch_size
        chunks = torch.chunk(inpt, n_chunks, dim=0)

        assert torch.isfinite(inpt).all(), "input has NaNs/Infs"
        assert inpt.device == next(self._model.parameters()).device, "input on wrong device"

        results = []
        with no_grad():
            for c in chunks:
                logits = self._model(c)
                output = self._softmax(logits) if self._apply_softmax else logits
                results.append(output.detach().cpu())
        return torch.cat(results, dim=0)
