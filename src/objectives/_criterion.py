import logging
from abc import ABC, abstractmethod
from typing import Any

from ._criteria_kwargs import criteria_kwargs


class Criterion(ABC):
    """A criterion, allowing to evaluate events."""

    _name: str
    _inverse: bool
    _allow_batched: bool

    def __init__(
        self, inverse: bool = False, allow_batched: bool = False
    ) -> None:  # TODO: remove defaults here, should be set in subclass.
        """
        Initialize the criterion.

        :param inverse: Whether the criterion should be inverted.
        :param allow_batched: Whether the criterion supports batching.
        """
        self._inverse = inverse
        self._allow_batched = allow_batched
        self._name = self._name + "Inv" if inverse else self._name

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> float:
        """
        Evaluate the criterion in question.

        :param kwargs: The KW-Args parsed.
        :returns: The value(s).
        """
        ...
        # TODO: maybe return tuples always

    @property
    def name(self) -> str:
        """
        Get the criterions name.

        :returns: The name.
        """
        return self._name

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Automatically apply wrapper if the evaluate function gets implemented.

        :param kwargs: The KW-Args parsed.
        :raises NotImplementedError: If the criterion does not support batching.
        """
        super().__init_subclass__(**kwargs)
        if "evaluate" in cls.__dict__:
            cls.evaluate = criteria_kwargs(cls.evaluate)
        orig_eval = cls.evaluate

        if "batch_dim" in kwargs and not cls._allow_batched:
            raise NotImplementedError(f"Criterion {cls.__name__} does not support batching.")

        def wrapped_evaluate(self, **kwargs):
            logging.info(f"Calculating fitness with {self._name}...")
            return orig_eval(self, **kwargs)

        cls.evaluate = wrapped_evaluate
