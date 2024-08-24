from typing import Dict, List, Union, Literal
from enum import Enum, auto

import numpy as np

from src.config import Config

class Resource(Enum):
    """Enumeration of resources that can be required by tasks.

    Inheritance from Enum allows for iteration over the values of the enumeration,
    and so it's hashable and can be used as a dictionary key.
    """
    drill = auto()
    centrifuge = auto()
    crane = auto()


class ExponentialDistribution:
    def __init__(self, lam: float):
        self.lam = lam
        if lam <= 0:
            raise ValueError("Lambda must be positive")
        self.scale = 1 / lam
        self.name = "Exponential"

    def realization(self) -> float:
        return np.random.exponential(self.scale)

    def __repr__(self):
        return f"Exp with lambda {self.lam}"

    def quantile(self, p: float) -> float:
        return -np.log(1-p)*self.scale

    @property
    def average(self) -> float:
        return self.scale

    @property
    def max(self) -> float:
        return self.quantile(Config.max_duration_quantile)


class IntProbabilityDistribution:
    tolerance = 1e-6

    def __init__(self, values: List[int], probabilities: List[float]):
        if any([p <= 0 for p in probabilities]):
            raise ValueError("Probabilities must be positive")
        if abs(sum(probabilities) - 1) > self.tolerance:
            raise ValueError("Probabilities must sum to 1")
        if len(values) != len(probabilities):
            raise ValueError("Values and probabilities must have same length")
        if len(values) != len(set(values)):
            raise ValueError("Values must be unique")
        if len(values) == 0:
            raise ValueError("Values and probabilities must be non-empty")
        for h in range(len(values)-1):
            if values[h] >= values[h+1]:
                raise ValueError("Values must be in increasing order")
        self._values = values
        self._probabilities = probabilities
        self._finish_prob_dict = self._prob_finish_at()
        self.name = "Int Discrete"

    @property
    def max(self) -> int:
        return max(self.values)

    def __repr__(self):
        return f"IntProbabilityDistribution with values {self.values} and probabilities {self.probabilities}"

    @property
    def values(self):
        return self._values

    @property
    def probabilities(self):
        return self._probabilities

    @values.setter
    def values(self, values):
        self._values = values

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    @property
    def average(self):
        return sum([value * probability for value, probability in zip(self.values, self.probabilities)])

    def quantile(self, p: float) -> int:
        if p < 0 or p > 1:
            raise ValueError(f"p must be between 0 and 1, not {p}")
        cumulative_probability = 0
        for value, probability in zip(self.values, self.probabilities):
            cumulative_probability += probability
            if cumulative_probability >= p:
                return value

    def realization(self) -> int:
        return np.random.choice(self.values, p=self.probabilities)

    def _prob_finish_at(self) -> Dict[int,float]:
        """Probability of finishing at time, given that it did not finish before."""
        finish_prob_dict: Dict[int,float] = {}
        remaining_prob: float = 1.
        for value, probability in list(zip(self.values, self.probabilities))[:-1]:
            finish_prob_dict[value]: float = probability / remaining_prob
            remaining_prob -= probability
        finish_prob_dict[self.values[-1]]: float = 1.
        return finish_prob_dict

    def prob_finish_at(self, time: int) -> float:
        return self._finish_prob_dict.get(time, 0.)


class Task:
    def __init__(
            self,
            id: int,
            resource_requirements: Dict[Resource, int],
            duration_distribution: IntProbabilityDistribution,
            dependencies: List[int]
    ):
        """Initialise a task with id, resource requirements, type of distribution for duration, and dependencies.

        :param id: Unique identifier for the task.
        :param resource_requirements: Dictionary with resources as keys and required amount of that resource as values.
        :param duration_distribution: Probabilistic distribution of the duration of the task.
        :param dependencies: List of task ids that must be completed before this task can start.
        """
        self.id: int = id
        self.dependencies: List[int] = sorted(dependencies)
        self.resource_requirements: Dict[Resource, int] = resource_requirements
        self.duration_distribution: Union[IntProbabilityDistribution,ExponentialDistribution] = duration_distribution
        self.minimal_dependencies: Union[None,List[int]] = None
        self.full_dependencies: Union[None,List[int]] = None

    def __repr__(self):
        return (f"Task {self.id} \n"
                f"with trimmed dependencies {self.minimal_dependencies}, \n"
                f"and full dependencies {self.full_dependencies}, \n"
                f"resource requirements {self.resource_requirements} \n"
                f"and duration {self.duration_distribution}\n"
                )

    def duration_realization(self) -> int:
        """Sample the distribution of the task and return a value with probability according to the distribution."""
        return self.duration_distribution.realization()

    def enough_resources(self, resources_available: Dict[Resource, int]) -> bool:
        """Check if the task can be executed with the available resources."""
        return all([resources_available[resource] >= self.resource_requirements[resource] for resource in self.resource_requirements])

    @property
    def average_duration(self) -> float:
        """Return the average duration of the distribution of the task."""
        return self.duration_distribution.average

    @property
    def minimal_duration(self) -> int:
        """Return the minimal duration of the distribution of the task.

        For exponential distributions, this is 0. (any continuous distribution where 0 is in the support)
        for discrete distributions, this is the smallest value that is explicitely in the support.
        """
        if isinstance(self.duration_distribution, ExponentialDistribution):
            return 0
        return min(self.duration_distribution.values)

    @property
    def maximal_duration(self) -> float:
        """Return the maximal duration of the distribution of the task.

        For exponential distributions, this is infinity. (any continuous distribution with infinite support).

        for discrete distributions, this is the largest value that is explicitely in the support."""
        if isinstance(self.duration_distribution, ExponentialDistribution):
            return np.inf
        return max(self.duration_distribution.values)

    def quantile_duration(self, p: float) -> float:
        """Return the p-quantile of the distribution of the task."""
        return self.duration_distribution.quantile(p)