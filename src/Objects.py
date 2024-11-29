from typing import Dict, List, Union, TypeVar, Literal, Tuple
from enum import Enum, auto
from abc import ABC, abstractmethod

import numpy as np

from src.utils import binom, MAX_DURATION_QUANTILE

T = TypeVar("T", bound="Task")


class Resource(Enum):
    """Enumeration of resources that can be required by tasks.

    Inheritance from Enum allows for iteration over the values of the enumeration,
    and so it's hashable and can be used as a dictionary key.
    """
    centrifuge = auto()
    crane = auto()
    drill = auto()


class ProbabilityDistribution(ABC):
    """Abstract Base class indicating what methods a probability distribution should have."""

    name: str

    def realization(self) -> Union[float,int]:
        """Sample from the (1d) distribution and return a single stochast."""
        pass

    def quantile(self, p: float) -> Union[float,int]:
        """Return the p-quantile of the distribution: the point on the indep (time) axis with a p-share of the
        surface area to its left."""
        pass

    @property
    @abstractmethod
    def average(self) -> float:
        """Return the expected duration of an event described by this distribution."""
        pass

    @property
    @abstractmethod
    def max(self) -> Union[float,int]:
        """Return the maximal duration of an event described by this distribution. I.e. worst case scenario."""
        pass


class ExponentialDistribution(ProbabilityDistribution):
    """Exponential distribution with parameter lambda. For describing the (continuous) duration between events in a
    Poisson process. Most of this repo is built on the assumption of memorylessness, so this is a key distribution."""

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
        return self.quantile(MAX_DURATION_QUANTILE)


class IntProbabilityDistribution(ProbabilityDistribution):
    """Toy prob distribution for tasks that can take a (small number of) integer durations.
    For instance, days or shifts. This makes the memory enumerable, and therefore discrete time evolution tractable.

    The size of the probabilities can be random, or can model binomial or uniform distributions.

    One simply moves through time-steps of the duration of the unit of this distribution, and there is a non-zero
    probability of finishing at each time-step.

    This class therefore also implements the probability of finishing at a certain time,
    given that it did not finish before: the probabilities must be renormalized to a sum of unity
    over all remaining time-steps.
    """
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
        self._finish_prob_dict = self._get_conditional_finish_probs()
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

    def _get_conditional_finish_probs(self) -> Dict[int, float]:
        """Pre-compute the probability of finishing at time, given that it did not finish before."""
        finish_prob_dict: Dict[int,float] = {}
        remaining_prob: float = 1.
        for value, probability in list(zip(self.values, self.probabilities))[:-1]:
            finish_prob_dict[value]: float = probability / remaining_prob
            remaining_prob -= probability
        finish_prob_dict[self.values[-1]]: float = 1.
        return finish_prob_dict

    def prob_finish_at(self, time: int) -> float:
        """Return the probability of finishing at time, given that it did not finish before.
        If queried on a time-step that is not in the support, return 0."""
        return self._finish_prob_dict.get(time, 0.)


class Task:
    """Fundamental unit of a project, with resource requirements, duration distribution, and dependencies."""

    MAX_STAGES = 8

    def __init__(
            self,
            task_id: int,
            resource_requirements: Dict[Resource, int],
            duration_distribution: ProbabilityDistribution,
            dependencies: List[int],
            stages: int = 1
    ):
        """Initialise a task with id, resource requirements, type of distribution for duration, and dependencies.

        :param task_id: Unique identifier for the task.
        :param resource_requirements: Dictionary with resources as keys and required amount of that resource as values.
        :param duration_distribution: Probabilistic distribution of the duration of the task.
        :param dependencies: List of task ids that must be completed before this task can start.
        :param stages: Number of times the duration of the task must pass before it is completed.
            Allows for modelling with an Erlang distribution, which does not have the memoryless property, and has
            A nonzero mode. Default is 1, which is the exponential distribution.
        """
        if stages < 1:
            raise ValueError(f"Task must have at least one stage. Got {stages}.")
        if stages > self.MAX_STAGES:
            raise ValueError(f"Task should have at most {self.MAX_STAGES} stages. Got {stages}.")

        self.id: int = task_id
        self.dependencies: List[int] = sorted(dependencies)
        self.resource_requirements: Dict[Resource, int] = resource_requirements
        self.duration_distribution: ProbabilityDistribution = duration_distribution
        self.minimal_dependencies: Union[None,List[int]] = None
        self.full_dependencies: Union[None,List[int]] = None
        self.stages: int = stages
        self.current_stage: int = 0

    def __repr__(self):
        return (f"Task {self.id} \n"
                f"with trimmed dependencies {self.minimal_dependencies}, \n"
                f"and full dependencies {self.full_dependencies}, \n"
                f"resource requirements {self.resource_requirements} \n"
                f"and duration {self.duration_distribution}\n"
                )

    def duration_realization(self) -> Union[int,float]:
        """Sample the distribution of the task and return a value with probability according to the distribution.

        This also progresses the task to the next stage."""
        if self.current_stage >= self.stages:
            raise ValueError("Task already finished")
        self.current_stage += 1
        return self.duration_distribution.realization()

    def enough_resources(self, resources_available: Dict[Resource, int]) -> bool:
        """Check if the task can be executed with the available resources."""
        return all([resources_available[resource] >= self.resource_requirements[resource] for resource in self.resource_requirements])

    @property
    def average_duration(self) -> float:
        """Return the average duration of the distribution of the task."""
        return self.duration_distribution.average*self.stages

    @property
    def minimal_duration(self) -> int:
        """Return the minimal duration of the distribution of the task.

        For exponential distributions, this is 0. (any continuous distribution where 0 is in the support)
        for discrete distributions, this is the smallest value that is explicitely in the support.
        """
        if isinstance(self.duration_distribution, ExponentialDistribution):
            return 0
        elif isinstance(self.duration_distribution, IntProbabilityDistribution):
            return min(self.duration_distribution.values)*self.stages

    @property
    def maximal_duration(self) -> Union[float,int]:
        """Return the maximal duration of the distribution of the task.

        For exponential distributions, this is infinity. (any continuous distribution with infinite support).

        for discrete distributions, this is the largest value that is explicitely in the support."""
        if isinstance(self.duration_distribution, ExponentialDistribution):
            return np.inf
        elif isinstance(self.duration_distribution, IntProbabilityDistribution):
            return max(self.duration_distribution.values)*self.stages

    def stage_quantile_duration(self, p: float) -> float:
        """Return the p-quantile of the distribution of the task."""
        return self.duration_distribution.quantile(p)

    @classmethod
    def generate_random(
            cls,
            task_id: int,
            dependencies: Union[List[int], None] = None,
            max_dependencies: int = 5,
            resource_types: Union[int, List[Resource]] = 2,
            max_simultaneous_resources_required: int = 10,
            minimum_duration_range: Tuple[int, int] = (1, 4),
            duration_spread_range: Tuple[int, int] = (1, 4),
            prob_type: Literal["uniform", "binomial", "random", "exponential"] = "exponential",
            max_stages: int = 5
    ) -> T:
        """
        Generate a task with random resource requirements and duration.

        :param task_id: integer denoting the task ids
        :param dependencies: list of task ids that this task depends on (must be done before)
            if none, will be randomly chosen from lower task ids
        :param max_dependencies: maximum number of dependencies randomly chosen
        :param resource_types: number of resource types or list of resource types
        :param max_simultaneous_resources_required: maximum number of resources required
        :param minimum_duration_range: range of minimum duration
        :param duration_spread_range: range of difference between maximum and minimum duration
            or if prob_type is exponential, the range from which to sample the average duration
        :param prob_type: type of probability distribution for duration
        :param max_stages: maximum number of stages for any task
        :return: task object
        """
        if isinstance(resource_types, int):
            resource_types: List[Resource] = list(np.random.choice(Resource, resource_types, replace=False))

        #error checks:
        if minimum_duration_range[1] < minimum_duration_range[0]:
            raise ValueError("Minimum duration range must be non-decreasing")
        if duration_spread_range[1] < duration_spread_range[0]:
            raise ValueError("Duration spread range must be non-decreasing")
        if minimum_duration_range[0] < 1:
            raise ValueError("Minimum duration must be at least 1")

        min_days = np.random.randint(minimum_duration_range[0], minimum_duration_range[1] + 1)
        max_days = min_days + np.random.randint(duration_spread_range[0], duration_spread_range[1] + 1)
        days = list(range(min_days, max_days + 1))
        stages = np.random.randint(1, max_stages + 1)

        if max_stages > cls.MAX_STAGES:
            raise ValueError(f"Stages must be at most {cls.MAX_STAGES}, not {max_stages}.")

        match prob_type:
            case "uniform":
                duration_distribution = IntProbabilityDistribution(
                    days,
                    [1 / (max_days - min_days + 1)] * (max_days - min_days + 1),
                )
            case "binomial":
                duration_distribution = IntProbabilityDistribution(
                    days,
                    [
                        binom(max_days - min_days, i) * 0.5 ** i * 0.5 ** (max_days - min_days - i) for i in
                        range(max_days - min_days + 1)
                    ],
                )
            case "random":
                probabilities: np.array = np.random.rand(max_days - min_days + 1)
                duration_distribution: IntProbabilityDistribution = IntProbabilityDistribution(
                    days, probabilities / sum(probabilities)
                )
            case "exponential":
                avg_duration = np.random.uniform(*duration_spread_range)
                duration_distribution: ExponentialDistribution = ExponentialDistribution(
                    1 / avg_duration)
            case other:
                raise ValueError(f"Probability type {other} not recognized")

        resource_requirements: Dict[Resource, int] = {
            resource_type: np.random.randint(1, max_simultaneous_resources_required + 1) for resource_type in
            resource_types
        }
        for resource in Resource:
            if resource not in resource_requirements:
                resource_requirements[resource] = 0
        if dependencies is None:
            if task_id > 0:
                dependencies: List[int] = list(np.random.choice(
                    range(task_id),
                    np.random.randint(0, min(task_id + 1, max_dependencies + 1)),
                    replace=False
                ))
            else:
                dependencies: List[int] = []

        return cls(
            task_id=task_id,
            resource_requirements=resource_requirements,
            duration_distribution=duration_distribution,
            dependencies=dependencies,
            stages=stages
        )

    @classmethod
    def from_dict(cls, task_dict: dict) -> T:
        """Create a task from a config dictionary."""

        if task_dict["distribution"].lower() in ["erlang", "exponential"]:
            probability_distribution = ExponentialDistribution(1/task_dict["avg_stage_duration"])
        else:
            raise ValueError(f"Memoryfull distributions currently not supported. Got {task_dict['distribution']}.")

        if task_dict["distribution"].lower() == "exponential" and task_dict["stages"] > 1:
            raise ValueError("Exponential distribution cannot have more than one stage. use 'erlang' instead.")

        resource_requirements = {
            Resource[resource]: amount for resource, amount in task_dict["resource_requirements"].items()
        }

        return cls(
            task_id=task_dict["id"],
            resource_requirements=resource_requirements,
            duration_distribution=probability_distribution,
            dependencies=task_dict["dependencies"],
            stages=task_dict["stages"]
        )


