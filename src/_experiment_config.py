from dataclasses import dataclass, field
from typing import Optional

from torch import nn

from src.criteria import Criterion


@dataclass
class ExperimentConfig:
    """A simple Dataclass to store experiment configs."""

    samples_per_class: int  # How candidates should be searched for per class
    generations: int  # How many generations we search for candidates.
    mix_dim_range: tuple[int, int]  # The range of mixing dimensions used.
    predictor: nn.Module  # The predictor network.
    generator: nn.Module  # The generator network.
    genome_size: int = field(init=False)  # The size of the genome.
    metrics: list[Criterion]  # The metrics used in search.
    classes: int  # The amount of classes in the experiment.
    save_to: Optional[str] = field(
        default=None
    )  # The name of the dataframe to save to, if None dont save.

    def __post_init__(self) -> None:
        # Calculate genome size from range of mixing.
        self.genome_size = self.mix_dim_range[1] - self.mix_dim_range[0]
