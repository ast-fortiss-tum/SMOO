from abc import ABC, abstractmethod
from typing import Any


class SUT(ABC):
    """An abstract system under test class."""

    @abstractmethod
    def process_input(self, inpt: Any) -> Any:
        """
        Process the input to the SUT.

        :param inpt: The input to process.
        :return: The processed input.
        """
        ...
