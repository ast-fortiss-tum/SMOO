from abc import ABC
from collections import UserList
from typing import Any, Generic, NoReturn, TypeVar, cast

TCandidate = TypeVar("TCandidate", bound="Candidate")


class Candidate(ABC):
    """An abstract class representing a solution candidate."""


class CandidateList(UserList[TCandidate], Generic[TCandidate]):
    """An abstract class representing a list of candidates."""

    def __init__(self, *initial_candidates: Candidate) -> None:
        """
        Initialize a CandidateList and the super class.

        :param initial_candidates: The initial candidates to populate with.
        """
        super().__init__(cast(list[TCandidate], list(initial_candidates)))

    """Make the list immutable."""

    def insert(self, index: Any = None, value: Any = None) -> NoReturn:
        """
        Empty insert function to make immutable.

        :param index: The index to insert.
        :param value: The value to insert.
        :raises TypeError: As the list is immutable.
        """
        raise TypeError()

    __setitem__ = insert
    __delitem__ = insert
    append = insert  # type: ignore[assignment]
    extend = insert  # type: ignore[assignment]
    pop = insert  # type: ignore[assignment]
    reverse = insert
    sort = insert  # type: ignore[assignment]
