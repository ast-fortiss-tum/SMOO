from typing import Any, Type, Union

from ._criterion import Criterion


class CriterionCollection:
    """A collection of criteria that simplifies calling and evaluation in addition to having better organization of results."""

    _criteria: list[Criterion]
    _results: dict[type[Criterion], Union[float, list[float]]]

    def __init__(self, criteria: list[Criterion]) -> None:
        """
        Initialize the collection of criteria.

        :param criteria: The criteria to add.
        """
        self._criteria = criteria
        self._results = dict()

    def evaluate_all(self, iargs: dict[str, Any]) -> None:
        """
        Evaluate all criteria in the collection.

        :param iargs: The input arguments.
        """
        for criterion in self._criteria:
            self._results[type(criterion)] = criterion.evaluate(**iargs)

    def get_results_of(
        self, criterion: Union[Type[Criterion], Criterion]
    ) -> Union[float, list[float]]:
        """
        Get results of a specific criterion type if available.

        :param criterion: The criterion type.
        :returns: The results.
        """
        key = criterion if isinstance(criterion, type) else type(criterion)
        return self._results[key]

    def get_all_results(self) -> dict[str, Union[float, list[float]]]:
        """
        Get all results of the collection in form of a dictionary.

        :returns: The results dictionary, with keys as criterion names.
        """
        return {c.name: self.get_results_of(c) for c in self._criteria}

    @property
    def names(self) -> list[str]:
        """
        Get the names of the criteria in the collection.

        :returns: The names of the criteria.
        """
        return [c.name for c in self._criteria]
