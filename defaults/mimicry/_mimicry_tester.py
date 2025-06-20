from __future__ import annotations

import logging
from datetime import datetime
from itertools import product
from typing import Optional

import numpy as np
import torch
import wandb
from numpy.typing import NDArray
from torch import Tensor
from wandb import UsageError

from src import SMOO
from src.manipulator import MixCandidate, MixCandidateList, StyleGANManipulator
from src.objectives import Criterion
from src.optimizer import Optimizer
from src.sut import ClassifierSUT

from ._default_df import DefaultDF
from ._experiment_config import ExperimentConfig


class MimicryTester(SMOO):
    """A tester class for DNN using latent space manipulation in generative models (mimicry)."""

    """Additional Parameters."""
    _num_w0: int
    _num_ws: int
    _config: ExperimentConfig

    """Temporary Variables."""
    _img_rgb: Tensor

    _manipulator: StyleGANManipulator

    def __init__(
        self,
        *,
        sut: ClassifierSUT,
        manipulator: StyleGANManipulator,
        optimizer: Optimizer,
        objectives: list[Criterion],
        config: ExperimentConfig,
        frontier_pairs: bool,
        num_w0: int = 1,
        num_ws: int = 1,
        silent_wandb: bool = False,
        restrict_classes: Optional[list[int]] = None,
    ):
        """
        Initialize the Neural Tester.

        :param sut: The system under test.
        :param manipulator: The manipulator object.
        :param optimizer: The optimizer object.
        :param objectives: The objectives list.
        :param config: The experiment config.
        :param frontier_pairs: Whether the frontier pairs should be searched for.
        :param num_w0: The number of primary seeds.
        :param num_ws: The number of target seeds.
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
            use_wandb=True,
        )

        self._num_w0 = num_w0
        self._num_ws = num_ws
        self._config = config

        self._df = DefaultDF(pairs=frontier_pairs, additional_fields=["genome"])

    def test(self, validity_domain: bool = False) -> None:
        """
        Testing the predictor for its decision boundary using a set of (test!) Inputs.

        :param validity_domain: Whether the validity domain should be tested for.
        """
        spc, c = self._config.samples_per_class, self._config.classes

        logging.info(
            f"Start testing. Number of classes: {len(c)}, iterations per class: {spc}, total iterations: {len(c)*spc}\n"
        )
        exp_start = datetime.now()  # Exp time for grouping
        for class_idx, sample_id in product(c, range(spc)):
            self._init_wandb(exp_start, class_idx, self._silent)  # Initialize Wandb run for logging

            iter_start = datetime.now()
            w0_tensors, w0_images, w0_ys, w0_trials = self._generate_seeds(self._num_w0, class_idx)

            """
            Now we select primary and secondary predictions for further style mixing.
            Note this can be extended to any n predictions, but for this approach we limited it to 2.
            Additionally this can be generalized to N w0 vectors, but now we only consider one.
            """
            _, second, *_ = torch.argsort(w0_ys[0], descending=True)[0]
            self._img_rgb = w0_images[0]

            wn_tensors, wn_images, wn_ys, wn_trials = (
                self._generate_noise(self._num_ws)
                if validity_domain
                else self._generate_seeds(self._num_ws, second)
            )

            self._maybe_log({"base_image": wandb.Image(self._img_rgb, caption="Base Image")})
            self._maybe_summary("w0_trials", wn_trials)
            self._maybe_summary("wn_trials", wn_trials)

            """
            Note that the w0s and ws' do not have to share a label, but for this implementation we do not control the labels separately.
            """
            # To save compute we parse tha cached tensors of w vectors as we generated them already for getting the initial prediction.
            w0c = [
                MixCandidate(label=class_idx, is_w0=True, w_tensor=tensor) for tensor in w0_tensors
            ]
            wsc = [MixCandidate(label=second.item(), w_tensor=tensor) for tensor in wn_tensors]
            candidates = MixCandidateList(*w0c, *wsc)

            # Now we run a search-based optimization strategy to find a good boundary candidate.
            logging.info(f"Running Search-Algorithm for {self._config.generations} generations.")
            for _ in range(self._config.generations):
                # We define the inner loop with its parameters.
                images, fitness, preds = self._inner_loop(candidates, class_idx, second)
                # Assign fitness to current population and additional data (in our case images).
                self._optimizer.assign_fitness(fitness, images, preds.tolist())
                self._optimizer.new_population()
            # Evaluate the last generation.
            images, fitness, preds = self._inner_loop(candidates, class_idx, second)
            self._optimizer.assign_fitness(fitness, images, preds.tolist())

            logging.info(
                f"\tBest candidate(s) have a fitness of: {', '.join([str(c.fitness) for c in self._optimizer.best_candidates])}"
            )
            self._maybe_summary("expected_boundary", second.item())
            wnb_results = {
                "best_candidates": wandb.Table(
                    columns=[metric.name for metric in self._objectives]
                    + [f"Genome_{i}" for i in range(self._optimizer.n_var)]
                    + ["Image"]
                    + [f"Conf_{i}" for i in self._restrict_classes],
                    data=[
                        [
                            *c.fitness,
                            *c.solution,
                            wandb.Image(c.data[0]),
                            *[c.data[1][i] for i in self._restrict_classes],
                        ]
                        for c in self._optimizer.best_candidates
                    ],
                ),
            }
            self._maybe_log(wnb_results)

            Xp, yp = self._optimizer.best_candidates[0].data
            genome = self._optimizer.best_candidates[0].solution
            results = [
                self._img_rgb.tolist(),
                w0_ys[0].tolist(),
                Xp.tolist(),
                yp,
                datetime.now() - iter_start,
                genome,
            ]
            self._df.append_row(results)
            self._optimizer.reset()  # Reset the learner for new candidate.
            logging.info("\tReset learner!")

        logging.info("Saving Experiments to DF")
        if self._config.save_to is not None:
            self._df.to_csv(f"{self._config.save_to}.csv", index=False)

    def _inner_loop(
        self,
        candidates: MixCandidateList,
        c1: int,
        c2: int,
    ) -> tuple[list[Tensor], tuple[NDArray, ...], Tensor]:
        """
        The inner loop for the learner.

        :param candidates: The mixing candidates to be used.
        :param c1: The base class label.
        :param c2: The second most likely label.
        :returns: The images generated, and the corresponding fitness and the softmax predictions.
        """
        # Get the initial population of style mixing conditions and weights
        sm_weights_arr = self._optimizer.get_x_current()
        sm_cond_arr = np.zeros_like(sm_weights_arr)
        assert (
            0 <= sm_cond_arr.max() < self._num_ws
        ), f"Error: StyleMixing Conditions reference indices of {sm_cond_arr.max()}, but we only have {self._num_ws} elements."

        images = []
        for sm_cond, sm_weights in zip(sm_cond_arr, sm_weights_arr):
            mixed_image = self._manipulator.manipulate(
                candidates=candidates,
                cond=sm_cond,
                weights=sm_weights,
                random_seed=self._get_time_seed(),
            )
            images.append(mixed_image)
        # Convert images to RGB if they are grayscale
        images = [self._assure_rgb(img) for img in images]

        """We predict the label from the mixed images."""
        predictions: Tensor = self._process(torch.stack(images))

        # TODO: maybe have candidates be the input for criterion evaluation.
        fitness = []
        for j, (Xp, yp) in enumerate(zip(images, predictions)):
            sol_arch = [i for i in images if not torch.equal(i, Xp)]
            gen_arch = [e for k, e in enumerate(sm_weights_arr) if k != j]
            im, lt, gt = [self._img_rgb, Xp], [c1, c2], sm_weights_arr[j]

            fitness.append(
                [
                    criterion.evaluate(
                        images=im,
                        logits=yp,
                        label_targets=lt,
                        genome_target=gt,
                        solution_archive=sol_arch,
                        genome_archive=gen_arch,
                    )
                    for criterion in self._objectives
                ]
            )
        fitness = tuple(map(np.array, zip(*fitness)))

        # Logging Operations
        results = {}
        # Log statistics for each objective function seperatly.
        for metric, obj in zip(self._objectives, fitness):
            results |= {
                f"min_{metric.name}": obj.min(),
                f"max_{metric.name}": obj.max(),
                f"mean_{metric.name}": obj.mean(),
                f"std_{metric.name}": obj.std(),
            }
        self._maybe_log(results)

        return images, fitness, predictions

    def _generate_seeds(
        self, amount: int, cls: int
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], int]:
        """
        Generate seeds for a specific class.

        :param amount: The amount of seeds to be generated.
        :param cls: The class to be generated.
        :returns: The w vectors generated, the corresponding images, confidence values and the amount of trials needed.
        """
        ws: list[Tensor] = []
        imgs: list[Tensor] = []
        y_hats: list[Tensor] = []

        logging.info(f"Generate seed(s) for class: {cls}.")
        # For logging purposes to see how many samples we need to find valid seed.
        trials = 0
        while len(ws) < amount:
            trials += 1
            # We generate w latent vector.
            w = self._manipulator.get_w(self._get_time_seed(), cls)
            # We generate and transform the image to RGB if it is in Grayscale.
            img = self._manipulator.get_image(w)
            img = self._assure_rgb(img)
            y_hat = self._process(img.unsqueeze(0))

            # We are only interested in candidate if the prediction matches the label
            if y_hat.argmax() == cls:
                ws.append(w)
                imgs.append(img)
                y_hats.append(y_hat)
        logging.info(f"\tFound {amount} valid seed(s) after: {trials} iterations.")
        return ws, imgs, y_hats, trials

    def _generate_noise(self, amount: int) -> tuple[list[Tensor], list[Tensor], list[Tensor], int]:
        """
        Generate noise.

        :param amount: The amount of seeds to be generated.
        :returns: The w vectors generated, the corresponding images, confidence values and the amount of trials needed.
        """
        logging.info("Generate noise seeds.")
        # For logging purposes to see how many samples we need to find valid seed.
        w: Tensor = self._manipulator.get_w(self._get_time_seed(), 0)
        ws = [torch.randn(w.size(), device=w.device) for _ in range(amount)]
        imgs = [self._assure_rgb(self._manipulator.get_image(w)) for w in ws]
        y_hats = [self._process(img.unsqueeze(0)) for img in imgs]

        logging.info(f"\tFound {amount} valid seed(s).")
        return ws, imgs, y_hats, 0

    def _init_wandb(self, exp_start: datetime, class_idx: int, silent: bool) -> None:
        """
        Initialize Wandb Run for logging

        :param exp_start: The start of the experiment (for grouping purposes).
        :param class_idx: The class index to search boundary candidates for.
        :param silent: Whether wandb should be silenced.
        """
        try:
            wandb.init(
                project="NeuralStyleSearch",
                config={
                    "num_gen": self._config.generations,
                    "num_w0s": self._num_w0,
                    "num_wns": self._num_ws,
                    "pop_size": self._optimizer._x_current.shape[0],
                    "experiment_start": exp_start,
                    "label": class_idx,
                    "learner_type": self._optimizer.optimizer_type,
                },
                settings=wandb.Settings(silent=silent),
            )
        except UsageError as e:
            logging.error(f"Raised error {e}, \n continuing...")
            pass
