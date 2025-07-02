from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """A simple Dataclass to store experiment configs."""

    samples_per_class: int  # How candidates should be searched for per class
    generations: int  # How many generations we search for candidates.

    classes: list[int]  # The classes in the experiment (-1 is unconditional).
    seeds: list[Optional[int]] = field(default_factory=list)  # Seeds for each sample.
    save_as: str = field(default="mimicry_exp")  # The name identifier for saved experiment files.

    def __post_init__(self) -> None:
        """Post processing for consistency."""
        if len(self.seeds) != self.samples_per_class:
            self.seeds = [None] * self.samples_per_class
