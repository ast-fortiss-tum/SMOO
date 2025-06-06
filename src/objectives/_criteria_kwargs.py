from typing import Callable, Union, _SpecialForm, get_args

import numpy as np
from torch import Tensor

DEFAULT_KWARGS = {
    "images": (list, Tensor),  # A list of images for the Criteria calculation.
    "solution_archive": (list, Tensor),  # An archive of other solutions for comparison.
    "logits": (Tensor,),  # Predicted class probabilities.
    "label_targets": (list, int),  # Could be [true label] or [primary label, secondary label, ...]
    "genome_target": (np.ndarray,),
    "genome_archive": (list, np.ndarray),
    "batch_dim": (Union[int, None],),  # Define batch dimension if evaluation is done batchwise.
}


def criteria_kwargs(func: Callable) -> Callable:
    """
    A decorator to enforce uniform Kwargs across criteria -> easier debugging.

    :param func: The function to wrap around.
    :returns: The wrapped function.
    """

    def wrapper(*args, **kwargs) -> Callable:
        """
        Checking kwargs and types.

        :param args: The arguments to check.
        :param kwargs: The keyword arguments to check.
        :returns: The parameterized function.
        :raises KeyError: If function has an invalid kwarg.
        :raises TypeError: If function has an invalid type for one of the kwargs.
        """
        ikw = set(kwargs.keys()) - DEFAULT_KWARGS.keys()
        if ikw:  # Invalid Keyword arguments.
            raise KeyError(f"Function {func.__name__} found invalid keyword arguments: {ikw}")

        for key, value in kwargs.items():
            expected_type = DEFAULT_KWARGS.get(key)
            elem_cond = True
            if hasattr(expected_type[0], "__iter__") and len(expected_type) > 1:
                elem_cond = isinstance(value, expected_type[0])
                elem_cond = elem_cond and all(isinstance(item, expected_type[1]) for item in value)
            elif isinstance(expected_type[0], _SpecialForm):
                elem_cond = isinstance(key, get_args(expected_type[0]))

            if not elem_cond:
                raise TypeError(
                    f"Argument {key} in function {func.__name__} must be of type {expected_type}, found {type(value)}.\n If it is an iterable check if all elements are of the expected type."
                )
        return func(*args, **kwargs)

    return wrapper
