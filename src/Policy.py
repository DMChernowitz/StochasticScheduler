from typing import List, Dict, Union, Tuple

from src.Project import Project
from src.Objects import Resource, Task
from src.StateSpace import State

class Policy:

    def __init__(self, project: Project, policy: List[int]):
        """Initialise a policy for a given project.

        :param project: The project to which the policy applies.
        :param policy: The policy, a list of task ids in order of priority.
        """
        self.policy: List[int] = policy
        self.original_policy: List[int] = policy.copy()
        self.project: Project = project

        # per time, list of tasks that are completed on that timestep
        self.future_task_finished: Dict[Union[int,float],List[int]] = {}

        # all tasks finished per time
        self.task_ids_finished_per_time: Dict[Union[int,float],List[int]] = {}

        # per time, list of tasks that (are planned to) start
        self.docket: Dict[Union[int,float],List[int]] = {}

        # per task id (dict key), whether the task is completed
        self.task_completion: Dict[int, bool] = {task_id: False for task_id in project.task_dict.keys()}

        # per resource type (dict key), the current occupation of the resource
        self.resource_available_current: Dict[Resource,int] = {
            resource: project.resource_capacities[resource]
            for resource in Resource
        }

        self.resources_snapshots: Dict[Union[int,float],Dict[Resource,int]] = {}
        self.state_sequence: List[Tuple[Union[int,float],State]] = []
        self.time_step: int = 0

    def execute(self):
        self.state_sequence = [(0,self.project.state_space.initial_state.copy())]
        self.time_step: float = 0.
        while not all(self.task_completion.values()):
            self.evolve()
        return self.time_step

    def evolve(self):
        """Execute one time step of the policy"""
        if self.future_task_finished != {}:
            # Move forward in time to the next task that finishes
            # and administer what happens when it finishes
            self.time_step: float = min(self.future_task_finished.keys())
            finished_task_ids: List[int] = self.future_task_finished.pop(self.time_step)
            for finished_task_id in finished_task_ids:
                self.task_ids_finished_per_time.setdefault(self.time_step,[]).append(finished_task_id)
                finished_task: Task = self.project.task_dict[finished_task_id]
                for resource, requirement in finished_task.resource_requirements.items():
                    self.resource_available_current[resource] += requirement
                self.task_completion[finished_task_id]: bool = True
                self.state_sequence.append((self.time_step,self.state_sequence[-1][1].finish_task(finished_task_id)))

        while (chosen_task_id := self.choose_task_id()) is not None:
            # choose a task to execute, and administer what happens when it starts, and set a time for it to finish
            chosen_task = self.project.task_dict[chosen_task_id]
            self.docket.setdefault(self.time_step,[]).append(chosen_task_id)
            chosen_task_finished = self.time_step + chosen_task.duration_realization()
            self.future_task_finished.setdefault(chosen_task_finished,[]).append(chosen_task_id)
            for resource, requirement in chosen_task.resource_requirements.items():
                self.resource_available_current[resource] -= requirement
            self.state_sequence.append((self.time_step,self.state_sequence[-1][1].start_task(chosen_task_id)))

        self.resources_snapshots[self.time_step]: Dict[Resource, int] = {
            resource: self.project.resource_capacities[resource] - self.resource_available_current[resource]
            for resource in Resource
        }

    def choose_task_id(self) -> Union[int,None]:
        """Main logic of a policy: which task to select. Highest ranked task that can be executed."""
        for j,task_id in enumerate(self.policy):
            if not self.task_completion[task_id]:
                task: Task = self.project.task_dict[task_id]
                prerequisites: List[bool] = [
                    self.task_completion[dependency] for dependency in task.minimal_dependencies
                ]
                if task.enough_resources(self.resource_available_current) and all(prerequisites):
                    # execute this task, and remove it from the policy (of to dos)
                    return self.policy.pop(j)
        return None

    def __repr__(self):
        return (
            "----------------------------------------\n"
            f"Policy with policy {self.original_policy} \n,"
            f"task_ids_finished_per_time {self.task_ids_finished_per_time} \n, "
            f"docket {self.docket} \n, "
            f"resources_used {self.resources_snapshots} \n"
            "----------------------------------------\n"
            )


class DynamicPolicy:
    """A policy that is dynamically updated based on the current state of the project."""
    def __init__(self, project: Project):
        self.project: Project = project
        # self.task_completion: Dict[int, bool] = {task_id: False for task_id in project.task_dict.keys()}
        # self.resource_available_current: Dict[Resource,int] = {
        #     resource: project.resource_capacities[resource]
        #     for resource in Resource
        # }
        self.time_step: Union[int,float] = 0
        self.current_state: State = self.project.state_space.initial_state.copy()
        self.state_sequence: List[Tuple[Union[int,float],State]] = [(self.time_step, self.current_state.copy())]

    def execute(self) -> Union[int,float]:
        """Execute the policy until the project is finished. Return the duration."""
        while not self.current_state.is_final:
            # get best estimate from dijkstra
            task_start_id = self.project.contingency_table[self.current_state]
            if task_start_id is None:
                # wait for a task to finish
                outcome: Dict[str, Union[float, State]] = self.project.state_space.wait_for_finish(self.current_state)
                self.time_step += outcome["time"]
                self.current_state: State = outcome["state"]
                self.state_sequence.append((self.time_step,self.current_state.copy()))
            else:
                # start a task
                self.current_state: State = self.current_state.start_task(task_start_id)
                self.state_sequence.append((self.time_step,self.current_state.copy()))
        return self.time_step





