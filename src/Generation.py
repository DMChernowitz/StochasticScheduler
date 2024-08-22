from typing import Literal, Union, List, Dict

from src.Objects import Task, Resource, IntProbabilityDistribution, ExponentialDistribution
from src.utils import binom

import numpy as np

def generate_task(
        task_id: int,
        dependencies: Union[List[int], None] = None,
        max_dependencies: int = 5,
        resource_types: Union[int,List[Resource]] = 2,
        max_simultaneous_resources_required: int = 10,
        min_days: int = 0,
        max_days: int = 10,
        prob_type: Literal["uniform", "binomial", "random"] = "uniform"
) -> Task:
    """
    Generate a task with random resource requirements and duration.

    :param task_id: integer denoting the task ids
    :param dependencies: list of task ids that this task depends on (must be done before)
        if none, will be randomly chosen from lower task ids
    :param max_dependencies: maximum number of dependencies randomly chosen
    :param resource_types: number of resource types or list of resource types
    :param max_simultaneous_resources_required: maximum number of resources required
    :param min_days: minimum duration in days
    :param max_days: maximum duration in days
    :param prob_type: type of probability distribution for duration
    :return: task object
    """
    if isinstance(resource_types, int):
        resource_types: List[Resource] = list(np.random.choice(Resource, resource_types, replace=False))

    days: List[int] = list(range(min_days, max_days + 1))

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
                    binom(max_days - min_days, i) * 0.5**i * 0.5**(max_days - min_days - i) for i in range(max_days - min_days + 1)
                ],
            )
        case "random":
            probabilities: np.array = np.random.rand(max_days - min_days + 1)
            duration_distribution: IntProbabilityDistribution = IntProbabilityDistribution(
                days, probabilities / sum(probabilities)
            )
        case "exponential":
            duration_distribution: ExponentialDistribution = ExponentialDistribution(1 / np.random.uniform(max_days - min_days + 1))
        case other:
            raise ValueError(f"Probability type {other} not recognized")

    resource_requirements: Dict[Resource, int] = {
        resource_type: np.random.randint(1, max_simultaneous_resources_required + 1) for resource_type in resource_types
    }
    for resource in Resource:
        if resource not in resource_requirements:
            resource_requirements[resource] = 0
    if dependencies is None:
        if task_id > 0:
            dependencies: List[int] = list(np.random.choice(
                range(task_id),
                np.random.randint(1, min(task_id+1,max_dependencies+1)),
                replace=False
            ))
        else:
            dependencies: List[int] = []

    return Task(task_id, resource_requirements, duration_distribution, dependencies)

