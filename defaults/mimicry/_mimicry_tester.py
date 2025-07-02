from __future__ import annotations

import json
import logging
import os
from itertools import product
from time import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor

from src import SMOO, TEarlyTermCallable
from src.manipulator import MixCandidate, MixCandidateList, StyleGANManipulator
from src.objectives import CriterionCollection
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
        objectives: CriterionCollection,
        config: ExperimentConfig,
        frontier_pairs: bool,
        num_w0: int = 1,
        num_ws: int = 1,
        restrict_classes: Optional[list[int]] = None,
        early_termination: Optional[TEarlyTermCallable] = None,
    ):
        """
        Initialize the Neural Tester.

        :param sut: The system-under-test.
        :param manipulator: The manipulator object.
        :param optimizer: The optimizer object.
        :param objectives: The objectives.
        :param config: The experiment config.
        :param frontier_pairs: Whether the frontier pairs should be searched for.
        :param num_w0: The number of primary seeds.
        :param num_ws: The number of target seeds.
        :param restrict_classes: What classes to restrict to.
        :param early_termination: An optional early termination function.
        """
        super().__init__(
            sut=sut,
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            restrict_classes=restrict_classes,
            use_wandb=False,
        )

        self._num_w0 = num_w0
        self._num_ws = num_ws
        self._config = config

        self._df = DefaultDF(pairs=frontier_pairs, additional_fields=["genome"])
        self._early_termination = early_termination or (lambda _: (False, None))
        self._term_early: bool = False

    def test(self, validity_domain: bool = False) -> None:
        """
        Testing the predictor for its decision boundary using a set of (test!) Inputs.

        :param validity_domain: Whether the validity domain should be tested for.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        spc, c, seeds = self._config.samples_per_class, self._config.classes, self._config.seeds

        logging.info(
            f"Start testing. Number of classes: {len(c)}, iterations per class: {spc}, total iterations: {len(c)*spc}\n"
        )

        for class_idx, sample_id in product(c, range(spc)):
            logging.info(f"Test class {class_idx}, sample idx {sample_id}.")
            start_time = time()  # Stores the start time of the current experiment.

            w0_tensors, w0_images, w0_ys, w0_trials = self._generate_seeds(
                self._num_w0, class_idx, seed=seeds[sample_id]
            )

            """
            Now we select primary and secondary predictions for further style mixing.
            Note this can be extended to any n predictions, but for this approach we limited it to 2.
            Additionally this can be generalized to N w0 vectors, but now we only consider one.
            """
            first, second, *_ = torch.argsort(w0_ys, descending=True)[0]
            self._img_rgb = w0_images[0]

            wn_tensors, wn_images, wn_ys, wn_trials = (
                self._generate_noise(self._num_ws)
                if validity_domain
                else self._generate_seeds(
                    self._num_ws, -1 if class_idx == -1 else second.item(), exclude=[first.item()]
                )
            )
            second, *_ = torch.argsort(wn_ys, descending=True)[0]

            """
            Note that the w0s and ws' do not have to share a label, but for this implementation we do not control the labels separately.
            """
            # We parse the cached tensors of w vectors as we generated them already for getting the initial prediction.
            w0c = [
                MixCandidate(label=first.item(), is_w0=True, w_tensor=tensor)
                for tensor in w0_tensors
            ]
            wsc = [MixCandidate(label=second.item(), w_tensor=tensor) for tensor in wn_tensors]
            candidates = MixCandidateList(*w0c, *wsc)

            # Track generation history for comprehensive logging
            all_gen_data: list[dict] = []
            budget_used: int = 0

            # Now we run a search-based optimization strategy to find a good boundary candidate.
            logging.info(f"Running Search-Algorithm for {self._config.generations} generations.")
            for gen in range(self._config.generations):
                gen_start = time()
                logging.info(f"Generation {gen + 1} start.")

                # We define the inner loop with its parameters.
                images, fitness, preds, term_cond, gen_data = self._inner_loop(
                    candidates, first.item(), second.item(), gen + 1
                )
                budget_used += images.shape[0]  # Add budget based on how many images are evaluated
                all_gen_data.append(gen_data)

                self._optimizer.assign_fitness(
                    fitness, [images[i] for i in range(images.shape[0])], preds.tolist()
                )

                logging.info(f"Generation {gen + 1} done in {time() - gen_start}.")
                if self._term_early:
                    logging.info(
                        f"Early termination triggered at generation {gen + 1} by {np.sum(term_cond)} individuals."
                    )
                    break
                # Assign fitness and additional data (in our case images) to the current population.
                self._optimizer.new_population()

            # Evaluate the last generation.
            if not self._term_early:
                images, fitness, preds, term_cond, gen_data = self._inner_loop(
                    candidates, first.item(), second.item(), self._config.generations
                )
                budget_used += images.shape[0]
                all_gen_data.append(gen_data)
                self._optimizer.assign_fitness(
                    fitness, [images[i] for i in range(images.shape[0])], preds.tolist()
                )

            """Save data."""
            log_dir = os.path.join(
                script_dir, f"runs/{self._config.save_as}_class_{first.item()}_{time()}"
            )
            os.makedirs(log_dir, exist_ok=True)

            # Save generation history as CSV
            df = pd.DataFrame(all_gen_data)
            df.to_csv(log_dir + "/data.csv", index=False)

            # Compile comprehensive stats
            stats = {
                "runtime": time() - start_time,
                "w0_trials": w0_trials,
                "wn_trials": wn_trials,
                "w0_predictions": w0_ys.cpu().squeeze().tolist(),
                "wn_predictions": wn_ys.cpu().squeeze().tolist(),
                "budget_used": budget_used,
                "expected_boundary": second.item(),
            }

            # Save best candidates and their data
            if self._term_early and term_cond is not None:
                """Here we save all elements that satisfy a termination condition."""
                indices = np.arange(term_cond.shape[0])[term_cond]
                for ind in indices:
                    self._save_tensor_as_image(images[ind], log_dir + f"/best_{ind}.png")
                    stats[f"best_{ind}_y_hat"] = preds[ind].tolist()
                    stats[f"best_{ind}_solution"] = self._optimizer.get_x_current()[ind].tolist()
                    stats[f"best_{ind}_fitness"] = [fitness[i][ind] for i in range(len(fitness))]
            else:
                """If no termination condition was met, we save the best candidates."""
                for i, bc in enumerate(self._optimizer.best_candidates):
                    self._save_tensor_as_image(bc.data[0], log_dir + f"/best_{i}.png")
                    stats[f"best_{i}_y_hat"] = bc.data[1]
                    stats[f"best_{i}_solution"] = bc.solution.tolist()
                    stats[f"best_{i}_fitness"] = list(bc.fitness)

            # Save origin and target images
            self._save_tensor_as_image(self._img_rgb, log_dir + f"/origin_{first.item()}.png")
            if wn_images.shape[0] > 0:
                self._save_tensor_as_image(wn_images[0], log_dir + f"/target_{second.item()}.png")

            # Save stats as JSON
            with open(f"{log_dir}/stats.json", "w") as f:
                json.dump(stats, f, indent=2)

            logging.info(
                f"\tBest candidate(s) have a fitness of: {', '.join([str(c.fitness) for c in self._optimizer.best_candidates])}"
            )

            # Keep the original DataFrame logging for backwards compatibility
            if not self._term_early:
                Xp, yp = self._optimizer.best_candidates[0].data
                genome = self._optimizer.best_candidates[0].solution
                results = [
                    self._img_rgb.tolist(),
                    w0_ys[0].tolist(),
                    Xp.tolist(),
                    yp,
                    time() - start_time,
                    genome,
                ]
                self._df.append_row(results)
            else:
                indices = np.arange(term_cond.shape[0])[term_cond]
                runtime = time() - start_time
                x_cur = self._optimizer.get_x_current()
                for ind in indices:
                    results = [
                        self._img_rgb.tolist(),
                        w0_ys[0].tolist(),
                        images[ind].tolist(),
                        preds[ind].tolist(),
                        runtime,
                        x_cur[ind],
                    ]
                    self._df.append_row(results)

            self._optimizer.reset()  # Reset the learner to have clean slate in next iteration.
            logging.info("\tReset learner!")

        logging.info("Saving Experiments to DF")
        if self._config.save_to is not None:
            self._df.to_csv(f"{self._config.save_to}.csv", index=False)

    def _inner_loop(
        self,
        candidates: MixCandidateList,
        c1: int,
        c2: int,
        generation: int,
    ) -> tuple[Tensor, tuple[NDArray, ...], Tensor, Optional[NDArray], dict]:
        """
        The inner loop for the learner.

        :param candidates: The mixing candidates to be used.
        :param c1: The base class label.
        :param c2: The second most likely label.
        :param generation: The current generation number.
        :returns: The images generated, the corresponding fitness, the softmax predictions, termination condition, and generation data.
        """
        # Get the initial population of style mixing conditions and weights
        sm_weights_arr = self._optimizer.get_x_current()
        sm_cond_arr = np.zeros_like(sm_weights_arr)
        assert (
            0 <= sm_cond_arr.max() < self._num_ws
        ), f"Error: StyleMixing Conditions reference indices of {sm_cond_arr.max()}, but we only have {self._num_ws} elements."

        # Generate all images in batch
        batch_images = self._manipulator.manipulate(
            candidates=candidates,
            cond=sm_cond_arr,
            weights=sm_weights_arr,
        )

        # Convert to batched tensor and ensure RGB
        images_tensor = self._assure_rgb(batch_images)

        """We predict the label from the mixed images."""
        predictions: Tensor = self._process(images_tensor)

        # Create origin batch for comparison
        origin_batch = self._img_rgb.expand(images_tensor.shape[0], *self._img_rgb.shape[1:])

        self._objectives.evaluate_all(
            {
                "images": [origin_batch, images_tensor],
                "logits": predictions,
                "label_targets": [c1, c2],
                "solution_archive": [],
                "batch_dim": 0,
            }
        )
        results = self._objectives.results
        fitness = tuple(np.asarray(f) for f in results.values())

        early_term, term_cond = self._early_termination(results)
        self._term_early = early_term
        if early_term:
            logging.info(f"Early termination condition met by: {term_cond.sum()} individuals")

        # Create generation data row for CSV logging
        gen_data = {
            "generation": generation,
        }
        gen_data |= results

        return images_tensor, fitness, predictions, term_cond, gen_data

    def _generate_seeds(
        self,
        amount: int,
        label: int,
        exclude: Optional[list[int]] = None,
        seed: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """
        Generate seeds for a specific class.

        :param amount: The number of seeds to be generated.
        :param label: The class to be generated.
        :param exclude: A list of classes to exclude.
        :param seed: A seed to be used for reproducibility.
        :returns: The generated w vectors, the corresponding images, confidence values and the amount of trials needed.
        """
        ws: list[Tensor] = []
        imgs: list[Tensor] = []
        y_hats: list[Tensor] = []

        logging.info(f"Generate seed(s) for class: {label}.")
        # For logging purposes to see how many samples we need to find valid seed.
        trials = 0
        while len(ws) < amount:
            trials += 1
            # We generate w latent vector.
            w = self._manipulator.get_w(self._get_time_seed() if seed is None else seed, label)
            # We generate and transform the image to RGB if it is in Grayscale.
            img = self._manipulator.get_images(w)
            img = self._assure_rgb(img)
            y_hat = self._process(img)

            # We are only interested in a candidate if the prediction matches the label.
            exclude_cond = y_hat.argmax().item() not in exclude if exclude is not None else True
            if (y_hat.argmax().item() == label) or (label == -1) and exclude_cond:
                ws.append(w)
                imgs.append(img)
                y_hats.append(y_hat)
        logging.info(f"\tFound {amount} valid seed(s) after: {trials} iterations.")

        # Convert lists to batched tensors
        ws_tensor = torch.stack(ws)
        images_tensor = torch.stack(imgs)
        y_hats_tensor = torch.cat(y_hats)

        return ws_tensor, images_tensor, y_hats_tensor, trials

    def _generate_noise(self, amount: int) -> tuple[Tensor, Tensor, Tensor, int]:
        """
        Generate noise.

        :param amount: The number of seeds to be generated.
        :returns: The generated w vectors, the corresponding images, confidence values and the number of trials needed.
        """
        logging.info("Generate noise seeds.")
        # For logging purposes to see how many samples we need to find valid seed.
        w: Tensor = self._manipulator.get_w(self._get_time_seed(), 0)
        ws = torch.randn((amount, *w.shape), device=w.device)
        images = self._assure_rgb(self._manipulator.get_images(ws))
        y_hats = self._process(images)

        logging.info(f"\tFound {amount} valid seed(s).")
        return ws, images, y_hats, 0
