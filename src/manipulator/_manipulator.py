from abc import ABC, abstractmethod
from typing import Any


class Manipulator(ABC):
    """An abstract manipulator class."""

    @abstractmethod
    def manipulate(self, candidates, cond, weights, random_seed) -> Any: ...
