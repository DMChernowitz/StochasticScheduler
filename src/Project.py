from typing import Dict, List, TypeVar, Type

from src.Objects import Task, Resource
from src.config import Config
from src.Generation import generate_task
from src.utils import prune_dependencies, format_table
from src.StateSpace import StateSpace

import networkx as nx

import numpy as np

T = TypeVar("T", bound="Project")


class Project:
    """A project is a storage class with some representation and self-checking of inputs.

    It holds information about tasks, resources, and the state space of the project.

    It is used by Policy and Experiment classes. They have methods that can carry out the project.
    """

    def __init__(self, tasks: List[Task], resource_capacities: Dict[Resource, int]):
        """Initialise a project with tasks and resource capacities.
        Also checks if the inputs are valid and constructs the state space.

        :param tasks: A list of tasks, using the Task class.
            Tasks must have unique ids and dependencies: ids of other Tasks.
        :param resource_capacities: A dictionary with resources as keys and capacities as values:
            the total amount of each resource available concurrently.
        """
        self.task_dict: Dict[int,Task] = {task.id: task for task in tasks}
        if not len({task.id for task in tasks}) == len(tasks):
            raise ValueError("Tasks must have unique ids")
        self.resource_capacities: Dict[Resource, int] = resource_capacities

        self.check_inputs()

        self.average_duration: float = sum([task.duration_distribution.average for task in tasks])
        self.state_space: StateSpace = StateSpace(tasks, resource_capacities)
        average_quantile = 1-1/np.exp(1)
        print(f"Creating project with {len(tasks)} tasks, "
              f"running stochastic Dijkstra's on it to find optimal contigent policy. ")
        self.contingency_table = self.state_space.construct_shortest_path_length(decision_quantile=average_quantile)

        print(f"Expected duration of fastest path: {round(self.state_space.expected_duration,4)}")
        print("(multiply by -log(1-p) for quantile p duration of each task, instead of expected.)")

    def visualize_state_space(self):
        self.state_space.visualize_graph()

    @property
    def max_time(self) -> int:
        """maximum time, if all tasks are performed in sequence (for exponentials, 0.999 quantile).

        Useful to upper bound the execution time of a policy.
        """
        return sum([task.duration_distribution.max for task in self.task_dict.values()])+1

    def check_inputs(self):
        """Check if the inputs are valid, raising a ValueError if not."""
        all_tasks = set(self.task_dict.keys())
        for task in self.task_dict.values():
            if not set(task.dependencies).issubset(all_tasks):
                missing_dependencies = set(task.dependencies) - all_tasks
                raise ValueError(f"Dependencies must be valid task ids. "
                                 f"Task {task.id} has dependencies {missing_dependencies} not in the project."
                                 )
        for resource in Resource:
            if resource not in self.resource_capacities:
                raise ValueError(f"Resource {resource} not in resource capacities")
        for task_id,task in self.task_dict.items():
            for resource, requirement in task.resource_requirements.items():
                if requirement > self.resource_capacities[resource]:
                    raise ValueError(f"Task {task_id} requires more {resource} than ever available")

    def __repr__(self):
        if len(self.task_dict) < 10:
            add_str: str = f"Number of topological orderings {self.n_topological_orderings}. \n"
        else:
            add_str: str = ""

        task_dict_array = [
            ["Task #", "Dependencies", "Distribution", "Avg duration", *[f"Req. {r.name}" for r in Resource]],
        ]
        for task in self.task_dict.values():
            task_dict_array.append([
                str(task.id),
                str(task.minimal_dependencies).strip("[").strip("]"),
                task.duration_distribution.name,
                str(round(task.duration_distribution.average,3)),
                *[str(task.resource_requirements.get(r,0)) for r in Resource]
            ])

        task_dict_repr = format_table(task_dict_array)

        resource_cap_str = ", ".join([f"{r.name}: {v}" for r,v in self.resource_capacities.items()])

        average_dur_str = str(round(self.average_duration,4))

        return (
            "-------------------------------------------- \n"
                f"Project with tasks: \n{task_dict_repr} \n"
                f"Resource capacities: ({resource_cap_str}), \n"
                f"Sequential duration expectation {average_dur_str}, "
                f"and state space of {len(self.state_space.states)} states.\n"
                f"{add_str}"
            "-------------------------------------------- \n"
        )

    @classmethod
    def from_config(cls, config: Config) -> T:
        """Create a project from a configuration object"""

        tasks: List[Task] = [
            generate_task(
                task_id=n,
                max_simultaneous_resources_required=config.max_simultaneous_resources_required,
                min_days=(a := np.random.randint(1, config.min_days_range)),
                max_days=a + np.random.randint(1, config.max_days_range),
                prob_type=config.prob_type
            ) for n in range(config.n_tasks)
        ]
        prune_dependencies(tasks)
        resource_capacities = {Resource(n): config.resource_available for n in range(1, 1 + len(Resource))}
        return cls(tasks, resource_capacities)

    @property
    def n_topological_orderings(self) -> int:
        """Return the number of topological orderings of the project's tasks.

        A proxy for the complexity of the project: In how many ways can the project be approached?

        Computationally expensive for large projects.
        """
        digraph = nx.DiGraph()
        digraph.add_edges_from(
            [(dep, task_id) for task_id, task in self.task_dict.items() for dep in task.dependencies]
        )
        return len(list(nx.all_topological_sorts(digraph)))