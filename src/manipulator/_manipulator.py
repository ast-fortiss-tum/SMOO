from abc import ABC, abstractmethod
from typing import Any

from ._candidate import Candidate, CandidateList


class Manipulator(ABC):
    """An abstract manipulator class."""

    @abstractmethod
    def manipulate(self, candidates: CandidateList[Candidate], **kwargs: Any) -> Any:
        """
        The manipulation function for the Manipulator.

        :param candidates: The candidates to manipulate.
        :param kwargs: Keyword arguments to pass to the manipulation function.
        :returns: The result of the manipulation.
        """
        ...
