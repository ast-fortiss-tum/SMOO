import gc
import logging
import os
from itertools import product
from time import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from torch import Tensor

from src import SMOO
from src.manipulator import DiffusionCandidate, DiffusionCandidateList, REPAEManipulator
from src.objectives import Criterion
from src.optimizer import Optimizer
from src.sut import ClassifierSUT


class DiffTester(SMOO):
    """A diffusion based tester."""

    _manipulator: REPAEManipulator
    _sut: ClassifierSUT

    def __init__(
        self,
        *,
        sut: ClassifierSUT,
        manipulator: REPAEManipulator,
        optimizer: Optimizer,
        objectives: list[Criterion],
        silent_wandb: bool = False,
        restrict_classes: Optional[list[int]] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize the Diffusion Tester.

        :param sut: The system under test.
        :param manipulator: The manipulator object.
        :param optimizer: The optimizer object.
        :param objectives: The objectives list.
        :param silent_wandb: Whether to silence wandb.
        :param restrict_classes: What classes to restrict to.
        :param use_wandb: Whether to use wandb for logging.
        """
        super().__init__(
            sut=sut,
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            silent_wandb=silent_wandb,
            restrict_classes=restrict_classes,
            use_wandb=use_wandb,
        )
        self._sut.set_device(self._manipulator._device)  # TODO: THis is ass

    def test(self):
        # TODO: the following should be config
        classes = [0]
        sample_per_class = 1
        generations = 25
        metric_names = [metric.name for metric in self._objectives]

        for class_id, sample_idx in product(classes, range(sample_per_class)):
            logging.info(f"Test class {class_id}, sample idx {sample_idx}.")
            log_dir = (
                f"runs/class_{class_id}_{time()}"  # TODO: for now exclude WANDB as it doesnt work?
            )
            os.makedirs(os.path.dirname(log_dir), exist_ok=True)

            while True:
                xt, emb = self._manipulator.get_diff_steps([class_id])
                origin_image = self._manipulator.get_image(xt[-1])
                y_hat = self._process(origin_image)
                if torch.argmax(y_hat) == class_id:
                    break
                logging.warning(
                    f"Failed to find initial candidate for {class_id}, predicted {torch.argmax(y_hat)}"
                )
            _, second, *_ = torch.argsort(y_hat[0], descending=True)

            while True:
                xtd, embd = self._manipulator.get_diff_steps([second])
                target_image = self._manipulator.get_image(xtd[-1])
                y_hatd = self._process(target_image)
                if torch.argmax(y_hatd) == second:
                    break
                logging.warning(
                    f"Failed to find target candidate for {second}, predicted {torch.argmax(y_hatd)}"
                )

            source, target = DiffusionCandidate(
                xt.squeeze(), emb, is_origin=True
            ), DiffusionCandidate(xtd.squeeze(), embd)
            candidates = DiffusionCandidateList(source, target)

            del xt, emb, xtd, embd
            self._cleanup()

            origin_batch = origin_image.expand(
                self._optimizer.get_x_current().shape[0], *origin_image.shape[1:]
            )  # These are just memory views: if inplace function needed do clone()

            all_gen_data = []
            for i in range(1, generations + 1):
                start = time()
                logging.info("_" * 50)
                logging.info(f"Generation {i} start.")
                weights = self._optimizer.get_x_current()

                xw = torch.as_tensor(weights[:, 0, :], device=self._manipulator._device)
                yw = torch.as_tensor(weights[:, 1, :], device=self._manipulator._device)
                xs_new = self._manipulator.manipulate(candidates, xw, yw)

                """We predict the label from the mixed images."""
                xs = self._manipulator.get_image(xs_new)
                predictions: Tensor = self._process(xs)

                logging.info("Calculating fitness...")
                fitness = [
                    c.evaluate(
                        images=[origin_batch, xs],
                        logits=predictions,
                        label_targets=[class_id, int(second.item())],
                        batch_dim=0,
                    )
                    for c in self._objectives
                ]

                row = {
                    "generation": i,
                    **{metric: vals for metric, vals in zip(metric_names, fitness)},
                }
                all_gen_data.append(row)
                xsc = np.ascontiguousarray(xs.cpu())
                self._optimizer.assign_fitness(fitness, xsc, predictions.numpy())
                self._optimizer.new_population()

                del xs, predictions, xw, yw, xsc, xs_new
                self._cleanup()  # Free up some memory
                logging.info(f"Generation {i} done in {time() - start}.")

            df = pd.DataFrame(all_gen_data)
            df.to_csv(log_dir + "/data.csv", index=False)

            Image.fromarray(self._optimizer.best_candidates.data[0]).save(log_dir + "/best.png")
            Image.fromarray(origin_image.squeeze().numpy()).save(log_dir + "/origin.png")
            Image.fromarray(target_image.squeeze().numpy()).save(log_dir + "/taget.png")

            logging.info(
                f"\tBest candidate(s) have a fitness of: {', '.join([str(c.fitness) for c in self._optimizer.best_candidates])}"
            )

    @staticmethod
    def _cleanup() -> None:
        """Cleanup memory stuff."""
        gc.collect()
        torch.cuda.empty_cache()

    def _init_wandb(self, exp_start: str, class_idx: int, silent: bool) -> None:
        """
        Initialize Wandb Run for logging

        :param exp_start: The start of the experiment (for grouping purposes).
        :param class_idx: The class index to search boundary candidates for.
        :param silent: Whether wandb should be silenced.
        """
        if self._use_wandb:
            try:
                wandb.init(
                    project="DiffManip",
                    config={
                        "experiment_start": exp_start,
                        "label": class_idx,
                        "learner_type": self._optimizer.learner_type,
                    },
                    settings=wandb.Settings(
                        silent=silent,
                        _disable_stats=True,
                        _disable_meta=True,
                    ),
                    reinit=True,
                    mode="online",
                )

            except wandb.errors.UsageError as e:
                logging.error(f"Raised error {e}, \n continuing...")
