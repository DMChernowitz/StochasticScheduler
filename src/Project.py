from typing import Dict, List, TypeVar, Set, Union

from src.Objects import Task, Resource
from config import Config, RandomConfig, LiteralConfig
from src.utils import format_table, DiGraph, EXPONENTIAL_AVERAGE_QUANTILE, VERBOSE, get_title_string
from src.StateSpace import StateSpace, State

T = TypeVar("T", bound="Project")


class Project:
    """A project is a storage class with some representation and self-checking of inputs.

    It holds information about tasks, resources, and the state space of the project.

    It is used by Policy and Experiment classes. They have methods that can carry out the project.
    """

    def __init__(self,
                 task_list: List[Task],
                 resource_capacities: Dict[Resource, int],
                 decision_quantile: float = EXPONENTIAL_AVERAGE_QUANTILE
                 ):
        """Initialise a project with tasks and resource capacities.
        Also checks if the inputs are valid and constructs the state space.

        :param task_list: A list of tasks, using the Task class.
            Tasks must have unique ids and dependencies: ids of other Tasks.
        :param resource_capacities: A dictionary with resources as keys and capacities as values:
            the total amount of each resource available concurrently.
        :param decision_quantile: The quantile of the distribution to use in the CSDP algorithm.
            default is 1-1/e, which in the exponential distribution coincides with the expected value.
        """
        self.task_list: List[Task] = sorted(task_list, key=lambda task: task.id)
        self.resource_capacities: Dict[Resource, int] = resource_capacities

        self._check_inputs()
        self.prune_dependencies()

        self.average_duration: float = sum([task.average_duration for task in task_list])
        self.state_space: StateSpace = StateSpace(task_list, resource_capacities)

        self.contingency_table = self.state_space.construct_contingency_table(decision_quantile=decision_quantile)

        if VERBOSE:
            print(f"Creating project with {len(task_list)} tasks, "
              f"running CSDP on it to find optimal contingent policy. ")
            print(f"Expected duration of fastest path: {round(self.state_space.expected_makespan, 4)}")

    def print_contingency_table(self) -> None:
        """Print the contingency table of the project."""
        start_task_list: List[List[State]] = [[] for _ in range(len(self.task_list))]
        for state, task_id in self.contingency_table.items():
            if task_id is not None:
                start_task_list[task_id].append(state)

        full_str = get_title_string("Contingency table")

        for h,task_starters in enumerate(start_task_list):
            full_str += f"Start task {h} at states {sorted(task_starters)}\n"
        full_str += "At all other states, no task can / should be started."
        print(full_str + get_title_string(""))

    def visualize_state_space(
            self,
            metastate_mode: bool = True,
            rich_annotations: bool = False,
            add_times: bool = False
    ) -> None:
        """Visualize the state space of the project as a graph.

        :param metastate_mode: If True, the graph will show metastates (states with the same tasks in progress).
        :param rich_annotations: If True, the graph will show the task ids of transitions, and the status of each task
            inside the state buttons. If False, only show the number of tasks in the metastate.
        :param add_times: If True, the graph will show the expected time to finish from each state, and the expected
            time to completion of each transition, if rich_annotations is also True. Only works outside metastate_mode.
        """
        self.state_space.visualize_graph(
            metastate_mode=metastate_mode,
            rich_annotations=rich_annotations,
            add_times=add_times
        )

    @property
    def max_makespan(self) -> Union[int, float]:
        """return maximum time, if all tasks are performed in sequence (for exponentials, 0.999 quantile).

        Useful to upper bound the execution time of a policy.
        """
        return sum([task.duration_distribution.max for task in self.task_list]) + 1

    def _check_inputs(self) -> None:
        """Check if the inputs are valid, raising a ValueError if not."""
        for h,task in enumerate(self.task_list):
            if task.id != h:
                raise ValueError(
                    "Tasks should have sequential task ids from 0 to [# of tasks-1], without gaps. "
                    f"However, the task with id {task.id} is at index {h}."
                )

        all_tasks = set(range(len(self.task_list)))
        for task in self.task_list:
            if not set(task.dependencies).issubset(all_tasks):
                missing_dependencies = set(task.dependencies) - all_tasks
                raise ValueError(f"Dependencies must be valid task ids. "
                                 f"Task {task.id} has dependencies {missing_dependencies} not in the project."
                                 )
            if any(task.id <= dependency for dependency in task.dependencies):
                raise ValueError(f"Task {task.id} has dependencies with higher ids than itself.")
        for resource in Resource:
            if resource not in self.resource_capacities:
                raise ValueError(f"Resource {resource} not in resource capacities")
        for task in self.task_list:
            for resource, requirement in task.resource_requirements.items():
                if requirement > self.resource_capacities[resource]:
                    raise ValueError(f"Task {task.id} requires more {resource} than ever available")

    def __repr__(self) -> str:
        """Return a string representation of the project."""
        if len(self.task_list) < 10:
            add_str: str = f"\n# topological orderings due to dependencies: {self.n_topological_orderings}."
        else:
            add_str: str = ""

        task_dict_array = [
            ["Task #", "Dependencies", "Distribution", "# stages", "Avg stage duration", *[f"Req. {r.name}" for r in Resource]],
        ]
        for task in self.task_list:
            task_dict_array.append([
                str(task.id),
                str(task.minimal_dependencies).strip("[").strip("]"),
                task.duration_distribution.name,
                str(task.stages),
                str(round(task.duration_distribution.average,3)),
                *[str(task.resource_requirements.get(r,0)) for r in Resource]
            ])

        task_dict_repr = format_table(task_dict_array)

        resource_cap_str = ", ".join([f"{r.name}: {v}" for r,v in self.resource_capacities.items()])

        average_dur_str = str(round(self.average_duration,4))

        return (
            get_title_string("Project") +
                f"With tasks: \n{task_dict_repr} \n"
                f"Resource capacities:             ({resource_cap_str}), \n"
                f"Sequential duration expectation: {average_dur_str}, \n"
                f"Optimal makespan expectation:    {round(self.state_space.expected_makespan, 4)}, \n"
                f"and state space of {len(self.state_space.states)} states."
                f"{add_str}" +
            get_title_string("")
        )

    @classmethod
    def from_config(cls, config: Config) -> T:
        """Create a project instance from a configuration object.

        :param config: A configuration object, either RandomConfig or LiteralConfig.
        must have the following attributes:
            - resource_capacities: dict or int
            - n_tasks: int (RandomConfig only)
            - max_dependencies: int (RandomConfig only)
            - max_simultaneous_resources_required: int (RandomConfig only)
            - minimum_duration_range: Tuple[int,int] (RandomConfig only)
            - duration_spread_range: Tuple[int,int] (RandomConfig only)
            - prob_type: str (RandomConfig only)
            - max_stages: int (RandomConfig only)
            - tasks: List[Dict] (LiteralConfig only)

        :return: A project instance, generated randomly or explicitly."""

        if isinstance(config.resource_capacities, int):
            resource_capacities = {Resource(n): config.resource_capacities for n in range(1, 1 + len(Resource))}
        elif isinstance(config.resource_capacities, dict):
            resource_capacities = {Resource[n]: v for n, v in config.resource_capacities.items()}
        else:
            raise ValueError("config.resource_capacities must be an int or a dict of resource names to capacities")

        if isinstance(config, RandomConfig):

            tasks: List[Task] = [
                Task.generate_random(
                    task_id=n,
                    max_simultaneous_resources_required=config.max_simultaneous_resources_required,
                    minimum_duration_range=config.minimum_duration_range,
                    duration_spread_range=config.duration_spread_range,
                    max_dependencies=config.max_dependencies,
                    prob_type=config.prob_type,
                    max_stages=config.max_stages
                ) for n in range(config.n_tasks)
            ]

        elif isinstance(config, LiteralConfig):
            tasks: List[Task] = [
                Task.from_dict(task_dict) for task_dict in config.tasks
            ]
        else:
            raise ValueError("config must be of type RandomConfig or LiteralConfig")
        return cls(tasks, resource_capacities)

    def reset_task_stages(self) -> T:
        """Reset the stages of all tasks to 0, so they can be re-run.

        :return: The reset project instance."""
        for task in self.task_list:
            task.current_stage = 0
        return self

    @property
    def n_topological_orderings(self) -> int:
        """Return the number of topological orderings of the project's tasks.

        A proxy for the complexity of the project: In how many ways can the project be approached?

        Computationally expensive for large projects.
        """
        full_edges = [(dep, task.id) for task in self.task_list for dep in task.full_dependencies]

        # would it be faster with the minimal amount of edges?
        # minimal_edges = [(dep, task.id) for task in self.task_list for dep in task.minimal_dependencies]

        mini_digraph = DiGraph(N=len(self.task_list), edges=full_edges)

        return mini_digraph.n_topological_orders

    def prune_dependencies(self) -> None:
        """Remove dependencies from tasks that are already dependencies of dependencies.

        This is done to remove redundancy in the project description.

        No return value, modifies the Task objects in place."""

        # first find all dependencies of a task, including any depth in the dependency tree
        full_dep_dict: Dict[int, List[int]] = {}
        for task in self.task_list:
            full_dep_dict[task.id] = task.dependencies.copy()
            for dependency in task.dependencies:
                for dependency_of_dependency in full_dep_dict[dependency]:
                    if dependency_of_dependency not in full_dep_dict[task.id]:
                        full_dep_dict[task.id].append(dependency_of_dependency)

        # then remove dependencies that are already dependencies of dependencies
        for task in self.task_list:
            task.full_dependencies = full_dep_dict[task.id]
            dependencies_of_dependencies: Set[int] = {
                dependency_of_dependency
                for dependency in task.dependencies
                for dependency_of_dependency in full_dep_dict[dependency]
            }
            task.minimal_dependencies = sorted(
                [dependency for dependency in task.dependencies if dependency not in dependencies_of_dependencies]
            )

        # dependencies
