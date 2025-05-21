from abc import ABC
from collections import UserList


class Candidate(ABC):
    """An abstract class representing a solution candidate."""


class CandidateList(UserList):
    """An abstract class representing a list of candidates."""

    def __init__(self, *initial_candidates: Candidate) -> None:
        """
        Initialize a CandidateList and the super class.

        :param initial_candidates: The initial candidates to populate with.
        """
        super().__init__(initial_candidates)

    """Make the list immutable."""

    def insert(self, index=None, value=None):
        raise TypeError()

    __setitem__ = insert
    __delitem__ = insert
    append = insert
    extend = insert
    pop = insert
    reverse = insert
    sort = insert
