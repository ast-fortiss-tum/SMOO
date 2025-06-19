from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from src.objectives import Criterion


def get_early_termination(
    target_criterion: Criterion,
    target_condition: Callable[[NDArray], NDArray],
    fulfill: str = "any",
) -> Callable[[dict], tuple[bool, Optional[NDArray]]]:
    """
    Get an early termination condition from provided parameters.

    :param target_criterion: The target criterion to check for.
    :param target_condition: The condition to evaluate the results with.
    :param fulfill: How the condition should be fulfilled. Either "any" or "all" (default: any).
    :return: The callable function.
    :raises ValueError: if fulfill is not "any" or "all".
    """
    if fulfill == "any":
        ff = np.any
    elif fulfill == "all":
        ff = np.all
    else:
        raise ValueError(f"Unknown fulfill option: {fulfill}")

    def condition_function(results: dict) -> tuple[bool, Optional[NDArray]]:
        """
        The condition generated.

        :param results: The result dictionary.
        :returns: The condition fulfillment and the condition values.
        """
        values = results.get(target_criterion.name)
        if values is None:
            return False, None
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)

        cond = target_condition(values)
        return bool(ff(cond)), cond

    return condition_function
