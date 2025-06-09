import gc
import json
import logging
import os
from itertools import product
from time import time
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import wandb
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from src import SMOO
from src.manipulator import DiffusionCandidate, DiffusionCandidateList, REPAEManipulator
from src.objectives import Criterion
from src.optimizer import Optimizer, PymooOptimizer
from src.sut import ClassifierSUT

from ._experiment_config import ExperimentConfig


class DiffTester(SMOO):
    """A diffusion based tester."""

    _manipulator: REPAEManipulator
    _optimizer: PymooOptimizer
    _sut: ClassifierSUT
    _config: ExperimentConfig

    def __init__(
        self,
        *,
        sut: ClassifierSUT,
        manipulator: REPAEManipulator,
        optimizer: Optimizer,
        objectives: list[Criterion],
        config: ExperimentConfig,
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
        :param config: The experiment config.
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
        self._config = config

    def test(self) -> None:
        """Start the diffusion based testing."""
        metric_names = [metric.name for metric in self._objectives]

        script_dir = os.path.dirname(os.path.abspath(__file__))
        for class_id, sample_idx in product(
            self._config.classes, range(self._config.samples_per_class)
        ):
            logging.info(f"Test class {class_id}, sample idx {sample_idx}.")

            log_dir = os.path.join(
                script_dir, f"runs/class_{class_id}_{self._config.save_as}_{time()}"
            )
            os.makedirs(log_dir, exist_ok=True)

            """Get initial origin and target candidates."""
            source, y_hat, origin_image = self._find_valid_candidate(class_id, is_origin=True)
            origin_batch = origin_image.expand(
                self._optimizer.get_x_current().shape[0], *origin_image.shape[1:]
            )  # These are just memory views: if inplace function needed do .clone()
            _, second, *_ = torch.argsort(y_hat[0], descending=True)

            target, _, target_image = self._find_valid_candidate(second, is_origin=False)
            candidates = DiffusionCandidateList(source, target)

            """Get the default solution shape for manipulation."""
            desired_solution_shape = self._optimizer.get_x_current().shape
            solution_cache = np.zeros(desired_solution_shape)
            """Start population based optimization."""
            all_gen_data, global_start = [], time()
            for f, solution_size in enumerate(self._config.optimizer_schedule):
                """Adapt problem to fit solution chunk."""
                self._optimizer.update_problem(solution_shape=(2, solution_size))
                logging.info("=" * 50)
                logging.info(
                    f"Optimizing solution chunk {f+1}/{len(self._config.optimizer_schedule)}"
                )
                intermediate_generations = self._config.generations // len(
                    self._config.optimizer_schedule
                )
                start_idx = 0
                for i in range(1, intermediate_generations + 1):
                    gen_start = time()
                    logging.info("_" * 50)
                    logging.info(f"Generation {i} start.")

                    solution_cache[:, :, start_idx : start_idx + solution_size] = (
                        self._optimizer.get_x_current()
                    )
                    xw = torch.as_tensor(solution_cache[:, 0, :], device=self._manipulator._device)
                    yw = torch.as_tensor(solution_cache[:, 1, :], device=self._manipulator._device)
                    xs_new = self._manipulator.manipulate(candidates, xw, yw)

                    """We predict the label from the mixed images."""
                    xs = self._manipulator.get_image(xs_new)
                    predictions: Tensor = self._process(xs)

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
                    logging.info(f"Generation {i} done in {time() - gen_start}.")

                """Assign best candidates uniformly to the cached solution."""
                stack = np.stack(self._optimizer.best_solutions_reshaped, axis=0)
                batch_size, num_cand = self._optimizer.get_x_current().shape[0], len(
                    self._optimizer.best_candidates
                )
                solutions = np.tile(stack, ((batch_size + num_cand - 1) // num_cand, 1, 1))[
                    :batch_size
                ]  # N x 2 x 50 -> B x 2 x 50

                solution_cache[:, :, start_idx : start_idx + solution_size] = solutions
                start_idx += solution_size

            """Save data."""
            stats = {"runtime": time() - global_start, "y_hat": y_hat.cpu().squeeze().tolist()}
            df = pd.DataFrame(all_gen_data)
            df.to_csv(log_dir + "/data.csv", index=False)

            for i, bc in enumerate(self._optimizer.best_candidates):
                image = bc.data[0]
                self._save_tensor_as_image(image, log_dir + f"/best_{i}.png")
                stats[f"best_{i}"] = bc.data[1].tolist()
                stats[f"best_{i}_solution"] = solution_cache[i].tolist()  # noqa

                tensor_img = torch.Tensor(image).unsqueeze(0)
                y_pred = self._sut.process_input(tensor_img)
                stats[f"best_{i}_y_hat"] = y_pred.squeeze().cpu().tolist()
            self._save_tensor_as_image(origin_image, log_dir + "/origin.png")
            self._save_tensor_as_image(target_image, log_dir + "/taget.png")

            with open(f"{log_dir}/stats.json", "w") as f:
                f.write(json.dumps(stats))

            logging.info(
                f"\tBest candidate(s) have a fitness of: {', '.join([str(c.fitness) for c in self._optimizer.best_candidates])}"
            )

    def _find_valid_candidate(
        self, class_id: int, is_origin: bool = False
    ) -> tuple[DiffusionCandidate, Tensor, Tensor]:
        """
        Sample single candidates that are valid to the SUT.

        :param class_id: The class ID.
        :param is_origin: Whether the candidate is a origin candidate.
        :returns: The DiffusionCandidate and the prediction of the SUT and the generated Image.
        """
        while True:
            xt, emb = self._manipulator.get_diff_steps([class_id])
            image = self._manipulator.get_image(xt[-1])
            y_hat = self._process(image)
            if torch.argmax(y_hat) == class_id:
                break
            logging.warning(
                f"Failed to find candidate for {class_id}, predicted {torch.argmax(y_hat)}"
            )
        candidate = DiffusionCandidate(xt.squeeze(), emb, is_origin=is_origin)
        return candidate, y_hat, image

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

    @staticmethod
    def _cleanup() -> None:
        """Cleanup memory stuff."""
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _save_tensor_as_image(tensor: Union[NDArray, Tensor], path: str) -> None:
        """
        Save a torch tensor [0,1] as an image.

        :param tensor: The tensor to save.
        :param path: The directory to save the image to.
        """
        array = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
        image = array.squeeze().transpose(1, 2, 0)  # C x H x W  -> H x W x C
        image = (image * 255).astype(np.uint8)  # [0,1] -> [0, 255]
        Image.fromarray(image).save(path)
