import datetime
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import wandb
from torch import Tensor

from .manipulator import Manipulator
from .objectives import Criterion
from .optimizer import Optimizer
from .sut import SUT


class SMOO(ABC):
    """A testing object based on the SMOO-Framework."""

    """Used Components."""
    _sut: SUT
    _manipulator: Manipulator
    _optimizer: Optimizer
    _objectives: list[Criterion]

    _restrict_classes: Optional[list[int]]
    _silent: bool

    def __init__(
        self,
        *,
        sut: SUT,
        manipulator: Manipulator,
        optimizer: Optimizer,
        objectives: list[Criterion],
        silent_wandb: bool,
        restrict_classes: Optional[list[int]],
    ):
        """
        Initialize the Neural Tester.

        :param sut: The system under test.
        :param manipulator: The manipulator object.
        :param optimizer: The optimizer object.
        :param objectives: The objectives list.
        :param silent_wandb: Whether to silence wandb.
        :param restrict_classes: What classes to restrict to.
        """

        self._sut = sut
        self._manipulator = manipulator
        self._optimizer = optimizer
        self._objectives = objectives

        self._silent = silent_wandb
        self._restrict_classes = restrict_classes

    @abstractmethod
    def test(self):
        """Every workflow needs a testing loop."""
        ...

    @staticmethod
    def _maybe_log(results: dict) -> None:
        """
        Logs to Wandb if initialized.

        :param results: The results to log.
        """
        try:
            wandb.log(results)
        except wandb.errors.Error as e:
            logging.error(e)
            pass

    @staticmethod
    def _maybe_summary(field: str, summary: Any) -> None:
        """
        Add elements to wandb Summary if initialized.

        :param field: The field to add an element to.
        :param summary: The element to add.
        """
        try:
            wandb.summary[field] = summary
        except wandb.errors.Error as e:
            logging.error(e)
            pass

    @staticmethod
    def _get_time_seed() -> int:
        """
        A simple function ot make a seed from the current timestamp.

        :returns: A seed based on the timestamp.
        """
        now = datetime.now()
        return int(round(now.timestamp()))

    @staticmethod
    def _assure_rgb(image: Tensor) -> Tensor:
        """
        Assure that image is or can be converted to RGB.

        :param image: The image to be converted.
        :returns: The converted image (3 x H x W).
        :raises ValueError: If the image shape is not recognized.
        """
        # We check if the input has a channel dimension.
        channel = None if len(image.shape) == 2 else len(image.shape) - 3
        # If we don`t have channels we add a dimension.
        image = image.unsqueeze(0) if channel is None else image

        rep_mask = [1] * len(image.shape)  # A repetition mask for channel extrusions
        if image.shape[channel] == 1:
            # If we only have one channel we repeat it 3 times to make it rgb.
            rep_mask[channel] = 3
            return image.repeat(*rep_mask)
        elif image.shape[channel] == 3:
            return image
        else:
            raise ValueError(f"Unknown image shape. {image.shape}")

    def _process(self, x: Tensor) -> Tensor:
        """
        A wrapper to predict with additional conditions.

        :param x: The image to be predicted.
        :returns: The predicted labels.
        """
        y_hat = self._sut.process_input(x)
        # TODO: restrict classes is too specific maybe refractor.
        return y_hat if self._restrict_classes is None else y_hat[:, self._restrict_classes]
