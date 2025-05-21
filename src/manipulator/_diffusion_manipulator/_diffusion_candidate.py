from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor

from .._candidate import Candidate, CandidateList


@dataclass
class DiffusionCandidate(Candidate):
    """A candidate solution for the diffusion based manipulator."""

    class_embedding: Union[list[Tensor], Tensor]  # The class embedding.
    xt: list[Tensor]  # The diffusion steps.

    def __post_init__(self) -> None:
        """Preprocessing of some elements after intialization."""
        if isinstance(self.class_embedding, Tensor):
            self.class_embedding = [self.class_embedding] * len(self.xt)


class DiffusionCandidateList(CandidateList):
    """A list of candidate solutions for the diffusion based manipulator."""

    _candidates: list[DiffusionCandidate]
    _class_embeddings: list[Tensor]
    _xts: list[Tensor]

    def __init__(self, *initial_candidates: DiffusionCandidate) -> None:
        """
        A candidate list for the diffusion based manipulator.

        :param initial_candidates: The initial candidate solutions used for manipulation.
        """
        super().__init__(*initial_candidates)
        self._candidates = list(initial_candidates)
        self._xts, self._class_embeddings = [], []
        for t in range(len(self._candidates[0].xt)):
            self._xts.append(torch.stack([c.xt[t] for c in self._candidates], dim=0))
            self._class_embeddings.append(
                torch.stack([c.class_embedding[t] for c in self._candidates], dim=0)
            )

    @property
    def class_embeddings(self) -> list[Tensor]:
        """
        Get the class embeddings for the entire candidate list as a single Tensor.

        :return: The class embeddings for the candidates.
        """
        return self._class_embeddings

    @property
    def xts(self) -> list[Tensor]:
        """
        Get a combined diffusion process for all candidates.

        :returns: The diffusion process for all candidates.
        """
        return self._xts

    def __getitem__(self, index: int) -> DiffusionCandidate:
        return self._candidates[index]

    @classmethod
    def from_diffusion_output(cls, xs: list[Tensor], emb: Tensor) -> DiffusionCandidateList:
        """
        Get a DiffusionCandidate list from a diffusion output.

        :param xs: The diffusion steps for the individual candidates.
        :param emb: The class embeddings for the diffusion candidates.
        :returns: A DiffusionCandidate list.
        """
        num_candidates = emb.shape[0]
        candidates = []
        for i in range(num_candidates):
            xt = [elem[i] for elem in xs]
            candidate = DiffusionCandidate(class_embedding=emb[i], xt=xt)
            candidates.append(candidate)
        return DiffusionCandidateList(*candidates)
