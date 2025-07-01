"""A package for latent space manipulators and auxiliary elements."""

from ._manipulator import Manipulator  # isort: skip

try:
    from ._diffusion_manipulator import (
        DiffusionCandidate,
        DiffusionCandidateList,
        REPAEManipulator,
    )
except:
    pass

try:
    from ._style_gan_manipulator import (
        MixCandidate,
        MixCandidateList,
        StyleGANManipulator,
    )
except:
    pass

__all__ = [
    "Manipulator",
    "MixCandidate",
    "MixCandidateList",
    "StyleGANManipulator",
    "REPAEManipulator",
    "DiffusionCandidate",
    "DiffusionCandidateList",
]
