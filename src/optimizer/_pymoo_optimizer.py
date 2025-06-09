import logging
from typing import Any, Type

import numpy as np
from numpy.typing import NDArray
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

from ._optimizer import Optimizer
from .auxiliary_components import OptimizerCandidate


class PymooOptimizer(Optimizer):
    """A Learner class for easy Pymoo integration"""

    _pymoo_algo: Algorithm
    _problem: Problem
    _pop_current: Population
    _bounds: tuple[int, int]
    _shape: tuple[int, ...]

    def __init__(
        self,
        bounds: tuple[int, int],
        algorithm: Type[Algorithm],
        algo_params: dict[str, Any],
        num_objectives: int,
        solution_shape: tuple[int, ...],
    ) -> None:
        """
        Initialize the genetic learner.

        :param bounds: Bounds for the optimizer.
        :param algorithm: The pymoo Algorithm.
        :param algo_params: Parameters for the pymoo Algorithm.
        :param num_objectives: The number of objectives the learner can handle.
        :param solution_shape: The shape of the solution arrays.
        """
        """Initialize Constants."""
        self._pymoo_algo = algorithm(**algo_params, save_history=True)
        self._optimizer_type = type(self._pymoo_algo)
        self._bounds = bounds
        self._num_objectives = num_objectives

        """Initialize optimization problem and initial solutions."""
        self.update_problem(solution_shape)

    def new_population(self) -> None:
        """
        Generate a new population.
        """
        logging.info("Sampling new population...")
        static = StaticProblem(self._problem, F=np.column_stack(self._fitness))
        Evaluator().eval(static, self._pop_current)
        self._pymoo_algo.tell(self._pop_current)

        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._normalize_to_bounds(self._pop_current.get("X"))

    def get_x_current(self) -> NDArray:
        """
        Return the current population in specific format.

        :return: The population as array of smx indices and smx weights.
        """
        return self._x_current.reshape((self._x_current.shape[0], *self._shape))

    def update_problem(self, solution_shape: tuple[int, ...]) -> None:
        """
        Change problem of optimization.

        :param solution_shape: The new solution shape.
        """
        lb, ub = self._bounds
        self._shape = solution_shape
        self._n_var = int(np.prod(solution_shape))

        self._problem = Problem(
            n_var=self._n_var, n_obj=self._num_objectives, xl=lb, xu=ub, vtype=float
        )
        self._pymoo_algo.setup(self._problem, termination=NoTermination())

        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._normalize_to_bounds(self._pop_current.get("X"))

        self._best_candidates = [
            OptimizerCandidate(
                solution=np.random.uniform(high=ub, low=lb, size=self._n_var),
                fitness=[np.inf] * self._num_objectives,
            )
        ]
        self._previous_best = self._best_candidates.copy()

    @property
    def best_solutions_reshaped(self) -> list[NDArray]:
        """
        Get the best solutions in correct shape.

        :return: The solutions.
        """
        return [c.solution.reshape(self._shape) for c in self._best_candidates]
