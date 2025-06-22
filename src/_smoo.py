import datetime
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import wandb
from torch import Tensor

from .manipulator import Manipulator
from .objectives import Criterion, CriterionCollection
from .optimizer import Optimizer
from .sut import SUT

TEarlyTermCallable = Optional[Callable[[Any], tuple[bool, Optional[Any]]]]


class SMOO(ABC):
    """A testing object based on the SMOO-Framework."""

    """Used Components."""
    _sut: SUT
    _manipulator: Manipulator
    _optimizer: Optimizer
    _objectives: Union[
        list[Criterion], CriterionCollection
    ]  # TODO: refractor stuff to remove lists

    _restrict_classes: Optional[list[int]]
    _silent: bool

    def __init__(
        self,
        *,
        sut: SUT,
        manipulator: Manipulator,
        optimizer: Optimizer,
        objectives: Union[list[Criterion], CriterionCollection],
        restrict_classes: Optional[list[int]],
        use_wandb: bool,
        early_termination: TEarlyTermCallable = None,
    ):
        """
        Initialize the SMOO Object.

        :param sut: The system-under-test.
        :param manipulator: The manipulator object.
        :param optimizer: The optimizer object.
        :param objectives: The objectives used.
        :param restrict_classes: What classes to restrict predictions to.
        :param use_wandb: Whether to use wandb.
        :param early_termination: A function that can be used to early terminate the workflow.
        """

        self._sut = sut
        self._manipulator = manipulator
        self._optimizer = optimizer
        self._objectives = objectives

        self._restrict_classes = restrict_classes
        self._use_wandb = use_wandb
        self._early_termination = early_termination or (lambda _: (False, None))

    @abstractmethod
    def test(self) -> None:
        """Every workflow needs a testing loop."""
        ...

    def _maybe_log(self, results: dict) -> None:
        """
        Logs to Wandb if initialized.

        :param results: The results to log.
        """
        if self._use_wandb:
            try:
                wandb.log(results)
            except wandb.errors.Error as e:
                logging.error(e)

    def _maybe_summary(self, field: str, summary: Any) -> None:
        """
        Add elements to wandb Summary if initialized.

        :param field: The field to add an element to.
        :param summary: The element to add.
        """
        if self._use_wandb:
            try:
                wandb.summary[field] = summary
            except wandb.errors.Error as e:
                logging.error(e)

    @staticmethod
    def _get_time_seed() -> int:
        """
        A simple function to generate a seed from the current timestamp.

        :returns: A seed based on the timestamp.
        """
        now = datetime.now()
        return int(round(now.timestamp()))

    @staticmethod
    def _assure_rgb(image: Tensor) -> Tensor:
        """
        Assure that an image is or can be converted to RGB.

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
        logging.info(f"Feeding Testcases to {self._sut.__class__.__name__}.")
        y_hat = self._sut.process_input(x)
        return y_hat if self._restrict_classes is None else y_hat[:, self._restrict_classes]
