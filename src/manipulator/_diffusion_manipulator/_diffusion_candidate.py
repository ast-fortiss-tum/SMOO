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
    is_origin: bool = False

    def __post_init__(self) -> None:
        """Preprocessing of some elements after intialization."""
        if isinstance(self.class_embedding, Tensor):
            self.class_embedding = torch.stack([self.class_embedding.squeeze()] * len(self.xt), dim=0)

class DiffusionCandidateList(CandidateList):
    """A list of candidate solutions for the diffusion based manipulator."""

    _candidates: list[DiffusionCandidate]
    _class_embeddings: Tensor
    _xts: Tensor  # N x  Diffusion Steps x Image Embedding
    _xts_origin: Tensor  # N_o x  Diffusion Steps x Image Embedding
    _xts_target: Tensor  # N_t x  Diffusion Steps x Image Embedding

    def __init__(self, *initial_candidates: DiffusionCandidate) -> None:
        """
        A candidate list for the diffusion based manipulator.

        :param initial_candidates: The initial candidate solutions used for manipulation.
        """
        super().__init__(*initial_candidates)
        self._candidates = list(initial_candidates)

        self._xts = torch.stack([c.xt for c in self._candidates], dim=0)
        self._xts_origin = torch.stack([c.xt for c in self._candidates if c.is_origin], dim=0)
        self._xts_target = torch.stack([c.xt for c in self._candidates if not c.is_origin], dim=0)

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

    @property
    def xts_origin(self) -> Tensor:
        """
        Get a combined diffusion process for origin candidates.

        The shape of the embeddings is as follows: Num origin candidates x Diffusion Steps x Latent Vectors

        :returns: The diffusion process for all candidates.
        """
        return self._xts_origin

    @property
    def xts_target(self) -> Tensor:
        """
        Get a combined diffusion process for target candidates.

        The shape of the embeddings is as follows: Num target candidates x Diffusion Steps x Latent Vectors

        :returns: The diffusion process for all candidates.
        """
        return self._xts_target


    def __getitem__(self, index: int) -> DiffusionCandidate:
        return self._candidates[index]

    @classmethod
    def from_diffusion_output(cls, xs: Tensor, emb: Tensor, are_origin: list[bool] = None) -> DiffusionCandidateList:
        """
        Get a DiffusionCandidate list from a diffusion output.

        :param xs: The diffusion steps for the individual candidates.
        :param emb: The class embeddings for the diffusion candidates.
        :param are_origin: Set origin seeds if applicable.
        :returns: A DiffusionCandidate list.
        """
        num_candidates = emb.shape[0]
        if are_origin is None:
            are_origin = [False] * num_candidates
        candidates = []
        for i, o in enumerate(are_origin):
            candidate = DiffusionCandidate(class_embedding=emb[i], xt=xs[:,i,...], is_origin=o)
            candidates.append(candidate)
        return DiffusionCandidateList(*candidates)
