from .._manipulator import Manipulator
from typing import Union
import torch

class DiffusionManipulator(Manipulator):
    """A manipulator based on a diffusion model."""

    _generator: torch.nn.Module
    _device: torch.device

    def __init__(
            self,
            generator: Union[torch.nn.Module, str],
            device: torch.device,
    ) -> None:
        """
        Initialize a diffusion manipulator.

        :param generator: Diffusion model to use.
        :param device: Device to use.
        """

    def manipulate(
            self,
            candidates: list,
            cond: list[int],
            weights: list[float],
            random_seed: int = 0,
    ) -> torch.Tensor:
        """
       Generate data using style mixing or interpolation.

       This function is heavily inspired by the Renderer class of the original StyleGANv3 codebase.

       :param candidates: The candidates used for style-mixing.
       :param cond: The manipulation conditions (layer combinations).
       :param weights: The weights for manipulating layers.
       :param random_seed: The seed for randomization.
       :returns: The generated image (C x H x W) as float with range [0, 255].
       """