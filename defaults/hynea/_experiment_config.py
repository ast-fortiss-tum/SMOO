from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """A dataclass to unify experimental configurations."""

    classes: list[int]  # A list of classes to test for.
    samples_per_class: int  # How many test cases are generated per class.
    generations: int  # How many generations to optimize for.
    optimizer_schedule: list[int]  # How the optimizer uses its generation budget.
    save_as: str  # just a string for better saving
    run_targeted: bool  # Whether targeted or untargeted approach is used.
    solution_shape: tuple[int, ...]  # The shape of the final solution (not partial solutions).

    restrict_classes: Optional[list[int]] = None  # Allows restricting classes considered in optim.
    use_diffusion_manipulation: bool = True  # Whether to use manipulations in diffusion.
    use_condition_manipulation: bool = True  # Whether to use manipulations in conditioning.
    reverse_schedule: bool = False  # Whether to reverse the solution schedule.

    def __post_init__(self) -> None:
        """Post-processing of the Experiment Config."""
        # If we use untargeted approach we don`t care about condition manipulation.
        self.use_condition_manipulation = self.use_condition_manipulation and self.run_targeted
