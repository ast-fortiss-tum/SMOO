from itertools import product
from typing import Optional

import torch
from torch import Tensor
import numpy as np
import logging
import wandb
from wandb.errors import UsageError

from src import SMOO
from src.manipulator import DiffusionCandidate, DiffusionCandidateList, REPAEManipulator
from src.objectives import Criterion
from src.optimizer import Optimizer
from src.sut import ClassifierSUT

from datetime import datetime


class DiffTester(SMOO):
    """A diffusion based tester."""

    _manipulator: REPAEManipulator
    _im_rgb: Tensor  # Temporary image storage

    def __init__(
        self,
        *,
        sut: ClassifierSUT,
        manipulator: REPAEManipulator,
        optimizer: Optimizer,
        objectives: list[Criterion],
        silent_wandb: bool = False,
        restrict_classes: Optional[list[int]] = None,
    ):
        """
        Initialize the Diffusion Tester.

        :param sut: The system under test.
        :param manipulator: The manipulator object.
        :param optimizer: The optimizer object.
        :param objectives: The objectives list.
        :param silent_wandb: Whether to silence wandb.
        :param restrict_classes: What classes to restrict to.
        """
        super().__init__(
            sut=sut,
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            silent_wandb=silent_wandb,
            restrict_classes=restrict_classes,
        )

    def test(self):
        # TODO: the follwoing should be config
        classes = list(range(10))
        sample_per_class = 10
        generations = 100

        exp_start = datetime.now()
        for class_id, sample_idx in product(classes, range(sample_per_class)):
            self._init_wandb(exp_start, class_id, self._silent)  # Initialize Wandb run for logging

            xt, emb = self._manipulator.get_diff_steps([class_id])
            image = self._manipulator.get_image(xt[-1])
            self._im_rgb = image.squeeze()
            y_hat = self._process(image)
            # TODO: check if SUT predicts correctly
            _, second, *_ = torch.argsort(y_hat[0], descending=True)

            target = DiffusionCandidate(*self._manipulator.get_diff_steps([second]))
            source = DiffusionCandidate(xt, emb)

            candidates = DiffusionCandidateList(source, target)

            for _ in range(generations):
                weights = self._optimizer.get_x_current()
                images = []
                for xw, yw in zip(weights[:,0,:], weights[:,1,:]): # TODO: this can be more efficient, batchwise
                    x_weights = torch.tensor(np.stack([xw, 1-xw], axis=1), device=self._manipulator._device)
                    y_weights = torch.tensor(np.stack([yw, 1 - yw], axis=1), device=self._manipulator._device)

                    x_new = self._manipulator.manipulate(candidates, x_weights, y_weights)
                    images.append(self._manipulator.get_image(x_new).squeeze())

                """We predict the label from the mixed images."""
                predictions: Tensor = self._process(torch.stack(images))

                fitness = []
                lt = [class_id, second]
                for j, (Xp, yp) in enumerate(zip(images, predictions)):  # TODO: this can be more efficient
                    ims = [self._im_rgb, Xp]
                    fitness.append(
                        [
                            c.evaluate(
                                images=ims,
                                logits=yp.cpu().numpy(),
                                label_targets=lt,
                            )
                            for c in self._objectives
                        ]
                    )
                fitness = tuple(map(np.array, zip(*fitness)))

                # Logging Operations
                results = {}
                # Log statistics for each objective function separately.
                for metric, obj in zip(self._objectives, fitness):
                    results |= {
                        f"min_{metric.name}": obj.min(),
                        f"max_{metric.name}": obj.max(),
                        f"mean_{metric.name}": obj.mean(),
                        f"std_{metric.name}": obj.std(),
                    }
                self._maybe_log(results)

                self._optimizer.assign_fitness(fitness, images, predictions.tolist())
                self._optimizer.new_population()

            logging.info(f"\tBest candidate(s) have a fitness of: {', '.join([str(c.fitness) for c in self._optimizer.best_candidates])}")
            self._maybe_summary("expected_boundary", second.item())
            wnb_results = {
                "best_candidates": wandb.Table(
                    columns=[metric.name for metric in self._objectives]
                            + [f"Genome_{i}" for i in range(self._optimizer.n_var)]
                            + ["Image"]
                            + [f"Conf_{i}" for i in range(10)],
                    data=[
                        [
                            *c.fitness,
                            *c.solution,
                            wandb.Image(c.data[0]),
                            *[c.data[1][i] for i in range(10)],
                        ]
                        for c in self._optimizer.best_candidates
                    ],
                ),
            }
            self._maybe_log(wnb_results)

    def _init_wandb(self, exp_start: datetime, class_idx: int, silent: bool) -> None:
        """
        Initialize Wandb Run for logging

        :param exp_start: The start of the experiment (for grouping purposes).
        :param class_idx: The class index to search boundary candidates for.
        :param silent: Whether wandb should be silenced.
        """
        try:
            wandb.init(
                project="DiffManip",
                config={
                    "experiment_start": exp_start,
                    "label": class_idx,
                    "learner_type": self._optimizer.learner_type,
                },
                settings=wandb.Settings(silent=silent),
            )
        except UsageError as e:
            logging.error(f"Raised error {e}, \n continuing...")
            pass
