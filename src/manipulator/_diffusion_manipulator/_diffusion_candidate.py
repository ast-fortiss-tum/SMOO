from __future__ import annotations

from dataclasses import dataclass
from typing import Union
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

    _origin: Union[DiffusionCandidateList, None] = None
    _target: Union[DiffusionCandidateList, None] = None

    def __init__(self, *initial_candidates: DiffusionCandidate, is_child: bool = False) -> None:
        """
        A candidate list for the diffusion based manipulator.

        :param initial_candidates: The initial candidate solutions used for manipulation.
        :param is_child: Whether the candidate list is a child of another candidate list.
        """
        super().__init__(*initial_candidates)
        self._candidates = list(initial_candidates)
        self._xts = torch.stack([c.xt for c in self._candidates], dim=0)

        if not is_child:
            self._origin = DiffusionCandidateList(*(c for c in self._candidates if c.is_origin), is_child=True)
            self._target = DiffusionCandidateList(*(c for c in self._candidates if not c.is_origin), is_child=True)

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
    def origin(self) -> Union[DiffusionCandidateList, None]:
        """
        Return origin diffusion candidates if applicable.

        :returns: The origin diffusion candidates.
        """
        return self._origin

    @property
    def target(self) -> Union[DiffusionCandidateList, None]:
        """
        Return target diffusion candidates if applicable.

        :returns: The origin diffusion candidates.
        """
        return self._target


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
