"""Hold the possible states in the state space of the project."""
from enum import Enum
from typing import List, Dict, Tuple, Union

import random

import numpy as np

from src.Objects import Task, Resource, ExponentialDistribution


class Stati(Enum):
    waiting = 0
    active = 1
    finished = 2


class State:
    """One state in the state space of the project."""

    allowed_values = [s.value for s in Stati]

    def __init__(self, stati: List[int]):
        for stat in stati:
            if stat not in self.allowed_values:
                raise ValueError(f"State must be one of {self.allowed_values}")
        self.stati: Tuple[int] = tuple(stati)

    def finish_task(self, task_id: int):
        if self.stati[task_id] != Stati.active.value:
            raise ValueError("task must be active to be finished")
        new_stati = list(self.stati)
        new_stati[task_id] = Stati.finished.value
        return State(new_stati)

    def start_task(self, task_id: int):
        if self.stati[task_id] != Stati.waiting.value:
            raise ValueError("task must be waiting to be started")
        new_stati = list(self.stati)
        new_stati[task_id] = Stati.active.value
        return State(new_stati)

    def copy(self):
        return State(list(self.stati))

    @property
    def is_initial(self):
        return all([self.stati[i] == Stati.waiting.value for i in range(len(self.stati))])

    @property
    def is_final(self):
        return all([self.stati[i] == Stati.finished.value for i in range(len(self.stati))])

    def __iter__(self):
        return iter(self.stati)

    def __getitem__(self, key):
        return self.stati[key]

    def __len__(self):
        return len(self.stati)

    def __hash__(self):
        return hash(self.stati)

    def __eq__(self, other):
        return self.stati == other.stati

    def __repr__(self):
        return "".join(map(str, self.stati))


class StateSpace:
    """Hold the possible states in the state space of the project."""

    def __init__(self, tasks: List[Task], resource_capacities: Dict[Resource, int]):
        self.tasks = tasks
        self.resource_capacities = resource_capacities

        self.graph: Dict[State, Dict[str, List[Tuple[int, State]]]] = self._graph_from_tasks()
        # transitions: can only be a single change, from waiting to active, or from active to finished
        # and from waiting to active, only dependent on the resources available
        # and contingent on dependencies being finished
        self.initial_state: Union[State, None] = None
        self.final_state: Union[State, None] = None
        for state in self.states:
            if state.is_initial:
                self.initial_state: State = state
            if state.is_final:
                self.final_state: State = state

        # initialize a hash table for the path lengths
        self.remaining_path_lengths: Dict[State, Union[None, Union[float, int]]] = {}
        # set the decision rule for timing:
        self.decision_quantile: Union[float, None] = None
        self.expected_duration: Union[float, None] = None
        self.contingency_table: Dict[State, Union[int, None]] = {}


    @property
    def states(self):
        return tuple(self.graph.keys())

    def descendants_of(self, state: State):
        return self.graph[state]["s"] + self.graph[state]["f"]

    def _graph_from_tasks(self) -> Dict[State, Dict[str, List[Tuple[int, State]]]]:
        for h, task in enumerate(self.tasks):
            if h != task.id:
                raise ValueError("Tasks must have ids equal to their index in the list")
        initial_state = State([Stati.waiting.value for _ in self.tasks])
        states = [initial_state]
        new_states = [initial_state]
        graph: Dict[State, Dict[str, List[Tuple[int, State]]]] = {}
        # now for each new state, get its descendants. If they are not in the list of states, add them
        # and add the transition to the graph
        while len(new_states) > 0:
            state = new_states.pop()
            descendants = self._get_descendants(state)
            graph[state] = descendants
            for index, descendant in descendants["s"] + descendants["f"]:
                if descendant not in states:
                    new_states.append(descendant)
                    states.append(descendant)
        return graph

    def _get_descendants(self, state: State) -> Dict[str, List[Tuple[int, State]]]:
        started: List[Tuple[int, State]] = []  # (task_id, state)
        finished: List[Tuple[int, State]] = []  # (task_id, state)
        currently_active: List[int] = [i for i in range(len(state)) if state[i] == Stati.active.value]
        resources_used: Dict[Resource, int] = {
            resource: sum([self.tasks[i].resource_requirements[resource] for i in currently_active]) for resource in
            Resource}
        resources_available: Dict[Resource, int] = {
            resource: self.resource_capacities[resource] - resources_used[resource] for resource in Resource}
        for h, j in enumerate(state):
            if j == Stati.active.value:
                finished.append((h, state.finish_task(h)))
            elif (
                    j == Stati.waiting.value
                    and
                    all([state[k] == Stati.finished.value for k in self.tasks[h].minimal_dependencies])
                    and
                    self.tasks[h].enough_resources(resources_available)
            ):
                started.append((h, state.start_task(h)))
        # can there be states that are allowed (for dependencies and resources),
        # but are still not reached in this branching?
        # no, because all possible orderings are explored.
        return {"s": started, "f": finished}

    def check_path_length(self):
        attempts = 1000
        path_lengths = []
        for _ in range(attempts):
            state = self.states[0]
            path_length = 0
            while state != State([Stati.finished.value for _ in state]) and path_length < 100:
                state = random.choice(self.descendants_of(state))[1]
                path_length += 1
            path_lengths.append(path_length)
        return path_lengths

    def construct_shortest_path_length(self, decision_quantile: float = 0.5) -> Dict[State, Union[int, None]]:
        """Perform first pass of stochastic dijkstra's algorithm
         to get the shortest expected path length to each state.

        Uses recursion, starting from the initial state, to find the expected duration to each state.
        This is done by adding the expected transition time to the expected duration of the next state.
        """
        self.remaining_path_lengths: Dict[State, Union[None, Union[float, int]]] = {
            self.final_state: 0
        }
        # The contingency table is the decision rule for each state: what to do next if we find ourselves in that state.
        self.contingency_table: Dict[State, Union[int, None]] = {self.final_state: None}
        self.decision_quantile = decision_quantile
        self.expected_duration = self.dynamic_step(self.initial_state)
        return self.contingency_table

    def dynamic_step(
            self,
            state,
    ):
        """Recursion step.

        Calculate the expected duration to reach a state, given the expected durations of its descendants.
        This is only implemented for exponential distributions, as the state space has no memory.

        If the path length to a state has already been calculated in a different branch, return it.

        Else, enumerate the possible transitions from the state, and calculate the expected duration to reach each

        There are two types of transitions:
        - starting a task: this takes no time, but the state changes
        - finishing a task: we must wait for the task to finish for the state to change.

        This method uses the state space graph to know what the possible transitions are from each state.
        """
        if state in self.remaining_path_lengths:
            # already calculated: escape now
            return self.remaining_path_lengths[state]

        # not yet calculated
        start_options: List[Tuple[Union[int, None], float]] = [(None, np.inf)]  # (task_id, time)
        for start_state in self.graph[state]["s"]:
            # starting a task takes no time
            start_options.append(
                (start_state[0], self.dynamic_step(start_state[1]))
            )
        best_start_option = min(start_options, key=lambda x: x[1])

        # There may be no active tasks to finish
        if self.graph[state]["f"]:

            finish_options, lambdas_options = self.get_wait_options(state)

            sum_lambda: float = sum(lambdas_options)
            # time until any task finishes is an exponential with the summed rate
            composite_exponential: ExponentialDistribution = ExponentialDistribution(sum_lambda)
            wait_time: float = composite_exponential.quantile(self.decision_quantile)

            # probability (ergo weight) of each task finishing first is proportional to its rate
            wait_option: float = sum(
                lam * (self.dynamic_step(option) + wait_time) for lam, option in zip(lambdas_options, finish_options)
            ) / sum_lambda
        else:
            # if there's no task to finish, we can wait forever
            wait_option = np.inf

        # there will always be a start option or a physical wait option,
        # or else we're in a final state, which doesn't reach this code
        if best_start_option[1] <= wait_option:
            # duration
            self.remaining_path_lengths[state]: Union[int, float] = best_start_option[1]
            # which task to start
            self.contingency_table[state]: int = best_start_option[0]
        else:
            self.remaining_path_lengths[state]: Union[int, float] = wait_option
            # don't start a task, but wait for one to finish
            self.contingency_table[state]: None = None
            if wait_option < np.inf and best_start_option[1] < np.inf:
                print(f"it was faster to wait for a task to finish than to start a new one, from {state}.")

        return self.remaining_path_lengths[state]

    def get_wait_options(self, state: State) -> Tuple[List[State],List[float]]:
        """return a list of possible states and the lambdas of their transitions."""
        if state not in self.graph:
            raise ValueError(f"State {state} not in state space")
        if not all([isinstance(self.tasks[task_id].duration_distribution, ExponentialDistribution) for task_id, _ in
                    self.graph[state]["f"]]):
            raise ValueError(f"State {state} has non-exponential tasks: not currently implemented")

        lambdas: List[float] = []
        next_states: List[State] = []

        for task_id, next_state in self.graph[state]["f"]:
            lambdas.append(self.tasks[task_id].duration_distribution.lam)
            next_states.append(next_state)
        return next_states, lambdas

    def wait_for_finish(self, state: State) -> Dict[str, Union[float, State]]:
        if not self.graph[state]["f"]:
            raise ValueError(f"State {state} has no active tasks")

        if len(self.graph[state]["f"]) == 1:
            task_id, next_state = self.graph[state]["f"][0]
            return {"time": self.tasks[task_id].duration_distribution.realization(), "state": next_state}

        next_states, lambdas = self.get_wait_options(state)

        sum_lambda: float = sum(lambdas)
        composite_exponential = ExponentialDistribution(sum_lambda)

        wait_time = composite_exponential.realization()
        new_state_n = np.random.choice(list(range(len(next_states))), p=[lam / sum_lambda for lam in lambdas])
        return {"time": wait_time, "state": next_states[new_state_n]}