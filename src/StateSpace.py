"""Hold the possible states in the state space of the project."""
from enum import Enum
from typing import List, Dict, Tuple, Union, TypeVar

import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import patches as mpatches

from src.Objects import Task, Resource, ExponentialDistribution

S = TypeVar("S", bound="State")

class Stati(Enum):
    waiting = 0
    active = 1
    finished = 2


class State:
    """One state in the state space of the project.

    Represented by concatenation of integers, each representing the state of a task at that index.
    The state of a task can be waiting, active, or finished, and the corresponding integers are 0, 1, and 2.
    """

    allowed_values = [s.value for s in Stati]

    def __init__(self, stati: List[int]):
        """Initialise a state with a list of integers representing the state of each task."""
        for stat in stati:
            if stat not in self.allowed_values:
                raise ValueError(f"State must be one of {self.allowed_values}")
        self.stati: Tuple[int] = tuple(stati)

    def finish_task(self, task_id: int) -> S:
        """Return the state that results from finishing a task at a given index."""
        if self.stati[task_id] != Stati.active.value:
            raise ValueError("task must be active to be finished")
        new_stati = list(self.stati)
        new_stati[task_id] = Stati.finished.value
        return State(new_stati)

    def start_task(self, task_id: int) -> S:
        """Return the state that results from starting a task at a given index."""
        if self.stati[task_id] != Stati.waiting.value:
            raise ValueError("task must be waiting to be started")
        new_stati = list(self.stati)
        new_stati[task_id] = Stati.active.value
        return State(new_stati)

    def copy(self):
        """Return a copy of the state."""
        return State(list(self.stati))

    @property
    def is_initial(self) -> bool:
        """Return True if the state is the initial state of the project, i.e. all tasks waiting to begin."""
        return all([self.stati[i] == Stati.waiting.value for i in range(len(self.stati))])

    @property
    def is_final(self) -> bool:
        """Return True if the state is the final state of the project, i.e. all tasks finished."""
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
        return "<"+"".join(map(str, self.stati))+">"

    @property
    def rank(self) -> int:
        """Return the depth of the state in the state space graph: number of vertices traversed to reach it."""
        return sum(self.stati)

class StateSpace:
    """Hold the possible states in the state space of the project.

    Also keeps track of the possible transitions between states, the graph topology, and the expected duration to reach
    each state.
    """

    def __init__(self, tasks: List[Task], resource_capacities: Dict[Resource, int]):
        """Initialise a state space with tasks and resource capacities.

        Also construct the state space graph, which is a dictionary of states,
            each with a dictionary of possible transitions.

        These are necessary because prerequisites and resource requirements determine the possible transitions
            and possible simultaneously active tasks.

        :param tasks: A list of tasks, using the Task class.
        :param resource_capacities: A dictionary with resources as keys and capacities as values
        """
        self.wait_is_faster_states = None  # States from which it is faster to wait for a task to finish
        # than to start a new one, despite resources being available: curious situation, worth keeping track

        self.tasks = tasks
        self.resource_capacities = resource_capacities

        self.graph: Dict[State, Dict[str, List[Tuple[int, State]]]] = self._graph_from_tasks()
        # transitions: can only be a single change, from waiting to active, or from active to finished
        # and from waiting to active, only dependent on the resources available
        # and contingent on dependencies being finished
        self.initial_state = State([Stati.waiting.value]*len(tasks))
        self.final_state = State([Stati.finished.value]*len(tasks))

        # initialize a hash table for the path lengths
        self.remaining_path_lengths: Dict[State, Union[None, Union[float, int]]] = {}
        # set the decision rule for timing:
        self.decision_quantile: Union[float, None] = None
        self.expected_duration: Union[float, None] = None
        self.contingency_table: Dict[State, Union[int, None]] = {}

    @property
    def states(self) -> Tuple[State]:
        """Return a tuple of all states in the state space."""
        return tuple(self.graph.keys())

    def descendants_of(self, state: State):
        """Return a list of possible transitions from a state, both due to starting and finishing tasks, after
        the graph has been constructed."""
        return self.graph[state]["s"] + self.graph[state]["f"]

    def _graph_from_tasks(self) -> Dict[State, Dict[str, List[Tuple[int, State]]]]:
        """Construct the state space graph from the tasks using recursion."""
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
        """Return the possible transitions from a state, both due to starting and finishing tasks.

        A transition is possible if the status of exactly one task is different, going from waiting to active,
        or from active to finished.
        Moreover, a task can only start if all its dependencies are finished,
        and if there are enough resources available for the task along with all other active tasks.
        Active tasks can always finish. This simply takes time, but that is not modelled here.

        :param state: The state from which to find the possible transitions.

        :return: A dictionary with two keys, "s" and "f", each with a list of tuples.
            The first element of the tuple is the task id that changes status
            The second element is the state that results from the transition.
        """
        # initialize result containers
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

        self.wait_is_faster_states = []  # reset the list of states where waiting is faster than starting a new task

        # by querying the initial state, we will recursively calculate the expected duration to reach all states
        self.expected_duration = self.dynamic_step(self.initial_state)
        if self.wait_is_faster_states:
            print("It was faster to wait for a task to finish than "
                  f"to start a new one from {len(self.wait_is_faster_states)} out of {len(self.states)} states.")
        else:
            print(f"This project has {len(self.states)} states "
                  "and it is always fastest to start at least one task when possible.")

        return self.contingency_table

    def dynamic_step(
            self,
            state,
    ):
        """Recursion step. Returns the expected duration to reach the final state from a given state.

        This duration depends on the state, the transition to its descendants, and the time from each descendant.
        Along the way, all durations from descendants are calculated and stored, recursively,
        in self.remaining_path_lengths

        This is only implemented for exponential distributions, as the state space has no memory.

        If the path length to a state has already been calculated in a different branch, it is returned immediately.

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
                self.wait_is_faster_states.append(state)

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

    def visualize_graph(self) -> None:
        """Create a graph of the graph."""

        state_ranks: Dict[int, List[State]] = {k: [] for k in range(2*len(self.tasks)+1)}

        for state in self.states:
            state_ranks[state.rank].append(state)

        # see how many states map to each state in the graph
        inverse_state_graph: Dict[State, List[State]] = {state: [] for state in self.states}
        for state, transitions in self.graph.items():
            for _, next_state in transitions["s"] + transitions["f"]:
                inverse_state_graph[next_state].append(state)

        def avg_pos_precursors(_state) -> float:
            if not inverse_state_graph[_state]:
                return 0
            return sum(state_positions[prev][1] for prev in inverse_state_graph[_state]) / len(inverse_state_graph[_state])

        # get position of each state on the canvas:
        state_positions: Dict[State, Tuple[int, int]] = {}
        for rank in state_ranks.keys():
            # heuristic: sort by the average height of the states that will point to this state
            # hopefully will make arrows as horizontal as possible
            states = sorted(
                state_ranks[rank],
                key=avg_pos_precursors
            )
            for i, state in enumerate(states):
                state_positions[state] = (rank, i)

        def make_arrow(start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> mpatches.Arrow:
            return mpatches.Arrow(start_pos[0], start_pos[1], 1, end_pos[1] - start_pos[1], width=0.3, alpha = 0.5)  # x, y, dx, dy

        # create the line collection data for the transitions
        sf_collections = {"s": [], "f": []}  # for the lines of task starting and task finishing
        for state, transitions in self.graph.items():
            for lab_letter in ["s", "f"]:
                for task_id, next_state in transitions[lab_letter]:
                    sf_collections[lab_letter].append(
                        make_arrow(state_positions[state], state_positions[next_state])
                    )

        fig, ax = plt.subplots()
        ax.set_xlim(-1, 2*len(self.tasks)+1)
        max_height = max(map(len, state_ranks.values()))
        ax.set_ylim(-1, max_height)

        if (lsc := len(sf_collections["s"])) != len(sf_collections["f"]):
            raise ValueError("There should be the same number of states starting and finishing in the total graph.")

        pc = PatchCollection(sf_collections["s"] + sf_collections["f"], color= ["r"]*lsc+["b"]*lsc,)

        # lc = LineCollection(start_collection+finish_collection, linewidths=1, colors=["r"]*lsc+["b"]*lsc)

        # plot the states as dots
        ax.scatter(*zip(*state_positions.values()), color="black")
        for label,if_state,col in [("Initial", self.initial_state, "g"), ("Final", self.final_state, "y")]:
            x,y = state_positions[if_state]
            ax.text(x,y+max_height/20, label[0], fontsize=12)
            ax.plot(x,y,marker="s",c=col, label=f"{label} State", markersize=10)


        ax.add_collection(pc)

        # access legend objects automatically created from data
        handles, _ = plt.gca().get_legend_handles_labels()

        # create manual symbols for legend
        handles.extend(
            [mpatches.Arrow(0,0,1,0,color=c, label=lab) for c, lab in [("r", "start task"), ("b", "finish task")]]
        )

        ax.legend(handles=handles)

        # x label
        ax.set_xlabel("Progress in project execution")
        ax.set_ylabel("Number of possible states")
        ax.set_title("State Space Transition Graph")

        plt.show()



