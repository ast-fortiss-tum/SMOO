from typing import Any

from sewar import uqi

from .._criterion import Criterion
from .._default_arguments import DefaultArguments
from ._prepare_tensor import prepare_tensor


class InvUQI(Criterion):
    """Implements the inverse universal image quality index using sewar."""

    def evaluate(self, *, default_args: DefaultArguments, **_: Any) -> float:
        """
        Get the inversed Universal Image Quality Index score.

        This score is in range (0,1) with 0 being the optimum.

        :param default_args: The default arguments parsed by the NeuralTester.
        :param _: Additional unused kwargs.
        :returns: The score.
        """
        i1, i2 = prepare_tensor(default_args.i1), prepare_tensor(default_args.i2)
        return 1 - uqi(i1, i2)
