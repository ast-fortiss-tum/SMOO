from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """A dataclass to unify experimental configurations."""

    classes: list[int]  # A list of classes to test for.
    samples_per_class: int  # How many test cases are generated per class.
    generations: int  # How many generations to optimize for.
    optimizer_schedule: list[int]  # How the optimizer uses its generation budget.
    save_as: str  # just a string for better saving
