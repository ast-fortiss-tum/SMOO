from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .._candidate import Candidate, CandidateList


@dataclass
class DiffusionCandidate(Candidate):
    """A candidate solution for the diffusion based manipulator."""

    xt: Tensor  # The diffusion steps (Diffusion Steps x Image Embedding).
    class_embedding: Tensor  # The class embedding (Diffusion Steps x Class Embedding).

    def __post_init__(self) -> None:
        """Preprocessing of some elements after intialization."""
        if isinstance(self.class_embedding, Tensor):
            self.class_embedding = torch.stack([self.class_embedding.squeeze()] * len(self.xt), dim=0)

class DiffusionCandidateList(CandidateList):
    """A list of candidate solutions for the diffusion based manipulator."""

    _candidates: list[DiffusionCandidate]
    _class_embeddings: Tensor
    _xts: Tensor  # N x  Diffusion Steps x Image Embedding

    def __init__(self, *initial_candidates: DiffusionCandidate) -> None:
        """
        A candidate list for the diffusion based manipulator.

        :param initial_candidates: The initial candidate solutions used for manipulation.
        """
        super().__init__(*initial_candidates)
        self._candidates = list(initial_candidates)
        self._xts = torch.stack([c.xt for c in self._candidates], dim=0)
        self._class_embeddings = torch.stack([c.class_embedding for c in self._candidates], dim=0)

    @property
    def class_embeddings(self) -> Tensor:
        """
        Get the class embeddings for the entire candidate list as a single Tensor.

        The shape of the embeddings is as follows: Num candidates x Diffusion Steps x Class Embeddings

        :return: The class embeddings for the candidates.
        """
        return self._class_embeddings

    @property
    def xts(self) -> Tensor:
        """
        Get a combined diffusion process for all candidates.

        The shape of the embeddings is as follows: Num candidates x Diffusion Steps x Latent Vectors

        :returns: The diffusion process for all candidates.
        """
        return self._xts

    def __getitem__(self, index: int) -> DiffusionCandidate:
        return self._candidates[index]

    @classmethod
    def from_diffusion_output(cls, xs: Tensor, emb: Tensor) -> DiffusionCandidateList:
        """
        Get a DiffusionCandidate list from a diffusion output.

        :param xs: The diffusion steps for the individual candidates.
        :param emb: The class embeddings for the diffusion candidates.
        :returns: A DiffusionCandidate list.
        """
        num_candidates = emb.shape[0]
        candidates = []
        for i in range(num_candidates):
            candidate = DiffusionCandidate(class_embedding=emb[i], xt=xs[:,i,...])
            candidates.append(candidate)
        return DiffusionCandidateList(*candidates)
