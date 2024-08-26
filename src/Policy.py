from typing import List, Dict, Union, Tuple

from src.Project import Project
from src.Objects import Resource, Task
from src.StateSpace import State

from src.utils import str_of_length

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
        self.task_completion: Dict[int, bool] = {task.id: False for task in project.task_list}

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
                finished_task: Task = self.project.task_list[finished_task_id]
                for resource, requirement in finished_task.resource_requirements.items():
                    self.resource_available_current[resource] += requirement
                self.task_completion[finished_task_id]: bool = True
                self.state_sequence.append((self.time_step,self.state_sequence[-1][1].finish_task(finished_task_id)))

        while (chosen_task_id := self.choose_task_id()) is not None:
            # choose a task to execute, and administer what happens when it starts, and set a time for it to finish
            chosen_task = self.project.task_list[chosen_task_id]
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
                task: Task = self.project.task_list[task_id]
                prerequisites: List[bool] = [
                    self.task_completion[dependency] for dependency in task.minimal_dependencies
                ]
                if task.enough_resources(self.resource_available_current) and all(prerequisites):
                    # execute this task, and remove it from the policy (of to dos)
                    return self.policy.pop(j)
        return None

    def get_gant_str(self, n_times: int =100) -> str:
        """Return a string representation of the gantt chart of the policy, after execution."""

        timescale = max(self.task_ids_finished_per_time.keys(), default=0)
        dt = timescale / n_times

        gantt_list = ["Task : |" + " " * n_times + "| (start, end ) | prereq: [ids]", "."*(n_times+39)]

        inverse_task_start_dict: Dict[int,Union[int,float]] = {}
        inverse_task_finish_dict: Dict[int,Union[int,float]] = {}

        for time,task_ids in self.task_ids_finished_per_time.items():
            for task_id in task_ids:
                inverse_task_finish_dict[task_id] = time

        for time,task_ids in self.docket.items():
            for task_id in task_ids:
                inverse_task_start_dict[task_id] = time

        for id in range(len(self.project.task_list)):

            start_time_str = str_of_length(inverse_task_start_dict.get(id,'?'),5)
            finish_time_str = str_of_length(inverse_task_finish_dict.get(id,'?'),5)

            prereq_str = str(self.project.task_list[id].minimal_dependencies)

            suffix = f"| ({start_time_str},{finish_time_str}) | prereq: {prereq_str}"
            task_str = str_of_length(id, 5)+": |"

            if id in inverse_task_start_dict and id in inverse_task_finish_dict:

                _s, _f = inverse_task_start_dict[id], inverse_task_finish_dict[id]

                middle_string = "".join(["■" if _s <= i * dt < _f else " " for i in range(n_times)])

            else:
                middle_string = " " * n_times

            gantt_list.append(task_str + middle_string + suffix)
        gantt_list.append(self.get_time_axis(n_times))

        return "\n".join(gantt_list)

    def get_time_axis(self, n_times: int) -> str:
        timescale = max(self.task_ids_finished_per_time.keys(), default=0)
        return "Time : 0 " + "." * (n_times - 4) + " " + str(timescale)[:5]

    def get_resource_chart(self, n_times: int = 100) -> str:
        """Return a string representation of the resource chart of the policy, after execution."""

        timescale = max(self.resources_snapshots.keys(), default=0)
        dt = timescale / n_times

        res_str = ""

        for resource in Resource:
            max_n_resources = self.project.resource_capacities[resource]
            resource_graph = [str_of_length(j,5)+": |" for j in range(max_n_resources)]

            # iterate over the snapshots of the resources. We are always in a certain period between two snapshots
            items_iter = iter(self.resources_snapshots.items())
            _, resource_dict = next(items_iter)
            next_snapshot, next_resource_dict = next(items_iter, (timescale,{}))
            # we want a column for each time step
            for i in range(n_times):
                while i * dt > next_snapshot:
                    resource_dict = next_resource_dict
                    next_snapshot, next_resource_dict = next(items_iter, (timescale,{}))
                for h_resource in range(max_n_resources):
                    if resource_dict[resource] > h_resource:
                        resource_graph[h_resource] += "■"
                    else:
                        resource_graph[h_resource] += " "
            res_str += f"Requirement of {resource}:\n"
            res_str += "\n".join(resource_graph[::-1]+[self.get_time_axis(n_times)])+"\n"
        return res_str

    def __repr__(self):

        if self.time_step == 0:
            return "Policy (not yet executed)"

        gant_str = self.get_gant_str()

        resource_str = self.get_resource_chart()

        return (
            "----------------------------------------\n"
            f"Policy with precedence {self.original_policy} \n"
            "Gantt Chart:\n"
            f"{gant_str}\n"
            f"{resource_str}\n"
            "----------------------------------------\n"
            )


class DynamicPolicy:
    """A policy that is dynamically updated based on the current state of the project."""
    def __init__(self, project: Project):
        self.project: Project = project
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





