import numpy as np
from numpy.typing import NDArray
from ._learner import Learner


class RevDELearner(Learner):
    """
    A Reverse Differential Evolutionary Learner.
    For theory see: https://arxiv.org/pdf/2002.02869.
    """
    # Parameters
    _population_size: int
    _bounds: tuple[int, int]
    F: float
    CR: float
    _continuous: bool

    # Population stuff
    _x_previous: NDArray
    _x_current_continuous: NDArray
    _fitness_current: NDArray

    def __init__(
            self,
            x0: NDArray,
            population_size: int,
            bounds:tuple[int, int] = (0,1),
            f: float = 0.9,
            cr: float = 0.5,
            continuous: bool = False,
    ) -> None:
        """
        Initialize the Reverse Differential Evolutionary Learner.

        :param x0: The initial genetic material.
        :param population_size: The size of the population.
        :param bounds: The bounds of the population.
        :param f: The scaling factor.
        :param cr: The crossover rate.
        :param continuous: If genomes are continuous or not.
        """
        self._generation = 0
        self._best_fitness = np.inf
        self._population_size = population_size

        self._bounds = bounds  # The bounds of genome values.
        self.F = f  # The scaling factor.
        self.CR = cr # The crossover rate.
        self._continuous = continuous

        self._x_current= x0
        self._fitness = np.empty(shape=x0.shape, dtype=float)

    def new_population(self, fitnesses: NDArray) -> None:
        """
        Generate a new population based on fitnesses of current population.

        :param fitnesses: The evaluated fitnesses.
        """
        x, f = self._select(self._x_current, fitnesses)
        x_cand, f_min = x[np.argmin(f)], np.min(f)
        self._best_candidate = (x_cand, f_min) if f_min < self._best_candidate[1] else self._best_candidate

        self._x_current = self._recombination(x)
        self._fitness = f

    def get_x_current(self) -> tuple[NDArray, NDArray]:
        """
        Return the current population in specific format.

        :return: The population as array of smx indices and smx weights.
        """
        smx_indices = self._x_current.round(0)
        smx_weights = self._x_current if self._continuous else np.ones_like(smx_indices)
        return smx_indices, smx_weights

    def _recombination(self, x: NDArray) -> NDArray:
        """
        Recombination method of genetic material.

        This approach discretizes the values since our genome is in N.

        :param x: The population to do recombination with.
        :return: The recombined population.
        """
        indices_1 = np.arange(x.shape[0])
        x_1 = x[indices_1]
        indices_2 = np.random.permutation(x.shape[0])
        x_2 = x_1[indices_2]
        indices_3 = np.random.permutation(x.shape[0])
        x_3 = x_2[indices_3]

        y_1 = np.clip(x_1 + self.F * (x_2 - x_3), *self._bounds)
        y_2 = np.clip(x_2 + self.F * (x_3 - y_1), *self._bounds)
        y_3 = np.clip(x_3 + self.F * (y_1 - y_2), *self._bounds)

        # uniform crossover
        if self.CR < 1.:
            p_1 = np.random.binomial(1, self.CR, y_1.shape)
            p_2 = np.random.binomial(1, self.CR, y_2.shape)
            p_3 = np.random.binomial(1, self.CR, y_3.shape)
            y_1 = p_1 * y_1 + (1. - p_1) * x_1
            y_2 = p_2 * y_2 + (1. - p_2) * x_2
            y_3 = p_3 * y_3 + (1. - p_3) * x_3

        x[[indices_1, indices_2, indices_3]] = y_1, y_2, y_3
        return x

    def _select(self, x: NDArray, f:NDArray) -> tuple[NDArray, NDArray]:
        """
        Select a subset of a population x by their fitness values.

        :param x: The population of candidates.
        :param f: The fitness values.
        :return: The selected population with its fitness.
        """
        indices = np.argsort(f)
        x_new, f_new = x[indices], f[indices]
        return x_new[:self._population_size], f_new[:self._population_size]
