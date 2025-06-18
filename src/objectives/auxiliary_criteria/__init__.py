"""A collection of criteria used for search based optimization methods."""

from ._archive_sparsity import ArchiveSparsity
from ._penalized_distance import PenalizedDistance

__all__ = [
    "PenalizedDistance",
    "ArchiveSparsity",
]
