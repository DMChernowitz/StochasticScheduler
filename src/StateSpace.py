"""Hold the possible states in the state space of the project."""
from typing import List, Dict, Tuple, Union, TypeVar, Iterable, Set

import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import patches as mpatches

from src.Objects import Task, Resource, ExponentialDistribution

S = TypeVar("S", bound="State")
MS = TypeVar("MS", bound="MetaState")


class State:
    """One state in the state space of the project.

    Represented by concatenation of integers, each representing the state of a task at that index.
    The state of a task can be waiting, active, or finished, and the corresponding integers are 0, 1, and 2.
    """

    finished = "f"
    # The extreme stages are 0 and f, all other stages are 'active' and have a rank of 1
    rank_mapping = {
        0: 0,
        "f": 2,
    }

    def __init__(
            self,
            total_stages: Iterable[int],
            current_stages: Iterable[Union[int,str]] = None,
            error_check: bool = True
    ):
        """Initialise a state of a project. It has an entry for each task in the project.

        The entries are numbers for the progress of the state.
        0 is waiting
        1-n is active, in the stages of a task with n stages
        f is finished.

        :param total_stages: The total stages per task that need to be traversed for the task to be finished,
            in order or task id
        :param current_stages: The current stage of each task, in order of task_id
        :param error_check: When constructing the state from outside, raise errors if params incorrectly configured.
            For speed, when a state produces another state, this can be skipped.
        """
        self.current_stages = tuple(current_stages)
        self.total_stages = tuple(total_stages)

        if error_check:
            self._error_check()

    def _error_check(self):
            if not (lts := len(self.total_stages)) == (lcs := len(self.current_stages)):
                raise ValueError(f"Require 1 current and 1 total stage per task. Got {lcs} current and {lts} total stages.")

            for current_stage, total_stage in zip(self.current_stages,self.total_stages):
                if not isinstance(total_stage, int) or not 0 < total_stage < 9:
                    raise ValueError("Total stages must be integers in [1,8]")
                if isinstance(current_stage, int):
                    if not 0 <= current_stage <= total_stage:
                        raise ValueError(f"Current stages must be in [0,total stage] or '{self.finished}'.")
                elif current_stage != self.finished:
                    raise ValueError(f"Non-integer current stages must be '{self.finished}' for finished tasks.")

    @classmethod
    def rank_from_stage(cls, stage: Union[int,str]):
        return cls.rank_mapping.get(stage, 1)

    def progress_task(self, task_id: int) -> S:
        """Return the state that results from progressing a task at a given index. Could be finishing it."""
        if self.current_stages[task_id] == self.finished:
            raise ValueError("Finished tasks cannot be progressed.")
        new_stati = list(self.current_stages)
        if new_stati[task_id] == self.total_stages[task_id]:
            new_stati[task_id] = self.finished
        else:
            new_stati[task_id] += 1
        return State(total_stages=self.total_stages, current_stages=new_stati, error_check=False)

    # def finish_task(self, task_id: int) -> S:
    #     """Return the state that results from finishing a task at a given index."""
    #     if self.stati[task_id] != Stati.active.value:
    #         raise ValueError("task must be active to be finished")
    #     new_stati = list(self.stati)
    #     new_stati[task_id] = Stati.finished.value
    #     return State(new_stati)
    #
    # def start_task(self, task_id: int) -> S:
    #     """Return the state that results from starting a task at a given index."""
    #     if self.stati[task_id] != Stati.waiting.value:
    #         raise ValueError("task must be waiting to be started")
    #     new_stati = list(self.stati)
    #     new_stati[task_id] = Stati.active.value
    #     return State(new_stati)

    def copy(self):
        """Return a copy of the state."""
        return State(self.total_stages, self.current_stages, error_check=False)

    @property
    def is_initial(self) -> bool:
        """Return True if the state is the initial state of the project, i.e. all tasks waiting to begin."""
        return all(c == 0 for c in self.current_stages)

    @property
    def is_final(self) -> bool:
        """Return True if the state is the final state of the project, i.e. all tasks finished."""
        return all(c == self.finished for c in self.current_stages)

    def task_complete(self, index) -> bool:
        """Return True if the task at index is finished."""
        return self.current_stages[index] == self.finished

    def __iter__(self):
        return iter(self.current_stages)

    def __getitem__(self, key):
        return self.current_stages[key]

    def __len__(self):
        return len(self.current_stages)

    def __hash__(self):
        return hash(self.current_stages)

    def __eq__(self, other):
        return self.current_stages == other.current_stages and self.total_stages == other.total_stages

    def __repr__(self):
        extremes = [0,self.finished]
        str_rep = [str(c) if c in extremes else f"{c}/{t}" for c, t in zip(self.current_stages, self.total_stages)]
        return "<"+"][".join(str_rep)+">"

    @property
    def rank(self) -> int:
        """Return the depth of the state in the state space graph: number of vertices traversed to reach it."""
        return sum(map(self.rank_from_stage, self.current_stages))

    def dependencies_finished(self, task: Task) -> bool:
        """Return True if all dependencies of a task are finished in this state."""
        return all(self[dep] == self.finished for dep in task.minimal_dependencies)

    def resources_used(self, task_list: List[Task]) -> Dict[Resource, int]:
        """Return the resources used by the active tasks in the state."""
        currently_active = [i for i, s in enumerate(self) if s not in self.rank_mapping]
        return {
            resource:
                sum(task_list[h].resource_requirements.get(resource, 0) for h in currently_active)
            for resource in Resource
        }


class StateSpace:
    """Hold the possible states in the state space of the project.

    Also keeps track of the possible transitions between states, the graph topology, and the expected duration to reach
    each state.
    """

    start = "s"
    finish = "f"
    progress = "p"
    transition_types = [start, finish, progress]

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

        total_stages = [task.stages for task in tasks]

        self.initial_state = State(total_stages=total_stages, current_stages=[0]*len(tasks))
        self.final_state = State(total_stages=total_stages, current_stages=[State.finished]*len(tasks))

        # transitions: can only be a single change, from waiting to active, or from active to finished
        # and from waiting to active, only dependent on the resources available
        # and contingent on dependencies being finished
        self.graph: Dict[State, Dict[str, List[Tuple[int, State]]]] = self._graph_from_tasks()

        # initialize a hash table for the path lengths
        self.remaining_path_lengths: Dict[State, Union[None, Union[float, int]]] = {}
        # set the decision rule for timing:
        self.decision_quantile: Union[float, None] = None
        self.expected_duration: Union[float, None] = None
        self.contingency_table: Dict[State, Union[int, None]] = {}
        self.metagraph: Dict[MetaState, Dict[str, List[Tuple[int, MetaState]]]] = {}
        self.states_per_metastate: Dict[MetaState, List[State]] = {}

    @property
    def states(self) -> Tuple[State]:
        """Return a tuple of all states in the state space."""
        return tuple(self.graph.keys())

    def descendants_of(self, state: State) -> List[Tuple[int, State]]:
        """Return a list of possible transitions from a state, both due to starting and finishing tasks, after
        the graph has been constructed."""
        return sum(self.graph[state].values(), [])

    def _graph_from_tasks(self) -> Dict[State, Dict[str, List[Tuple[int, State]]]]:
        """Construct the state space graph from the tasks using recursion."""
        for h, task in enumerate(self.tasks):
            if h != task.id:
                raise ValueError("Tasks must have ids equal to their index in the list")

        states = [self.initial_state]
        to_do_states = [self.initial_state]
        graph: Dict[State, Dict[str, List[Tuple[int, State]]]] = {}
        # now for each new state, get its descendants. If they are not in the list of states, add them
        # and add the transition to the graph
        while len(to_do_states) > 0:
            state = to_do_states.pop()  # take the next state on the docket
            descendants = self._get_descendants(state)
            graph[state] = descendants
            for index, descendant in sum(descendants.values(), []):
                if descendant not in states:
                    # all states are added to to_do_states exactly once, and
                    # are removed from it as we work through the list
                    to_do_states.append(descendant)
                    states.append(descendant)
        return graph

    def _resources_available(self, state: State) -> Dict[Resource, int]:
        """Return the resources available in a state."""
        resources_used = state.resources_used(self.tasks)
        return {
            resource: self.resource_capacities[resource] - resources_used[resource] for resource in Resource
        }

    def _get_descendants(self, state: State) -> Dict[str, List[Tuple[int, State]]]:
        """Return the possible transitions from a state, both due to starting and finishing tasks.

        A transition is possible if the status of exactly one task is different, going from waiting to active,
        progressing to the next stage, or from active to finished.
        Moreover, a task can only start if all its dependencies are finished,
        and if there are enough resources available for the task along with all other active tasks.
        Active tasks can always progress, and can thus finish. This simply takes time, but that is not modelled here.

        :param state: The state from which to find the possible transitions.

        :return: A dictionary with three keys, "s", "p" and "f", each with a list of tuples.
            The first element of the tuple is the task id that changes status/stage.
            The second element is the state that results from the transition.
        """
        extreme_stages = [0, State.finished]

        # initialize result containers
        result_containers: Dict[str, List[Tuple[int, State]]] = {t: [] for t in self.transition_types}
        # started/progressed/finished: List[Tuple[int, State]] = []  # (task_id, state)

        resources_available: Dict[Resource, int] = self._resources_available(state)
        for h, j in enumerate(state):
            if (
                    j == 0  # task is waiting
                    and
                    state.dependencies_finished(self.tasks[h])  # all dependencies are finished
                    and
                    self.tasks[h].enough_resources(resources_available)  # enough resources available
            ):
                result_containers[self.start].append((h, state.progress_task(h)))
            elif j not in extreme_stages:  # task is active
                next_state = state.progress_task(h)
                if next_state[h] == State.finished:  # was in the final stage
                    result_containers[self.finish].append((h, next_state))
                else:  # was in an intermediate stage
                    result_containers[self.progress].append((h, next_state))
        # can there be states that are allowed (for dependencies and resources),
        # but are still not reached in this branching?
        # no, because all possible orderings are explored.
        return result_containers

    def check_path_length(self):
        attempts = 1000
        path_lengths = []
        for _ in range(attempts):
            state = self.states[0]
            path_length = 0
            while state != self.final_state and path_length < 1000:
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

        self.wait_is_faster_states = []  # reset list of states from which waiting is faster than starting a new task

        if not all([isinstance(task.duration_distribution, ExponentialDistribution) for task in self.tasks]):
            raise ValueError(f"Project has non-exponential tasks: Dijkstra not currently implemented")

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

        This is only implemented for exponential/erlang distributions, as the state space has no memory.

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

        # initialize the duration to reach this state
        start_options: List[Tuple[Union[int, None], float]] = [(None, np.inf)]  # (task_id, time)
        for start_state in self.graph[state][self.start]:
            # starting a task takes no time
            start_options.append(
                (start_state[0], self.dynamic_step(start_state[1]))
            )
        best_start_option = min(start_options, key=lambda x: x[1])

        finish_options, lambdas_options, composite_exponential = self.get_wait_options(state)

        if lambdas_options:  # There may be active tasks to finish
            # time until any task finishes is an exponential with the summed rate
            wait_time: float = composite_exponential.quantile(self.decision_quantile)

            # probability (ergo weight) of each task finishing first is proportional to its rate
            # and the expected time is then the sum over expected times contingent on each task finishing first
            # times the probability of that task finishing first
            wait_option: float = wait_time + sum(
                lam * self.dynamic_step(option) for lam, option in zip(lambdas_options, finish_options)
            ) / composite_exponential.lam
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

    def get_wait_options(self, state: State) -> Tuple[List[State],List[float], ExponentialDistribution]:
        """return a list of possible states and the lambdas of their transitions."""
        if state not in self.graph:
            raise ValueError(f"State {state} not in state space")

        progress_and_finish = self.graph[state][self.progress] + self.graph[state][self.finish]

        if not progress_and_finish:  # no active tasks
            return [], [], ExponentialDistribution(1)

        lambdas, next_states = zip(
            *[(self.tasks[task_id].duration_distribution.lam, next_state)
              for task_id, next_state in progress_and_finish])

        composite_exponential = ExponentialDistribution(sum(lambdas))

        return next_states, lambdas, composite_exponential

    def wait_for_finish(self, state: State) -> Dict[str, Union[float, State]]:
        """Simulate waiting for a task to finish and return the time and the state that results from it."""

        next_states, lambdas, composite_exponential = self.get_wait_options(state)
        if not lambdas:
            raise ValueError(f"State {state} has no active tasks")

        wait_time = composite_exponential.realization()
        new_state_n = np.random.choice(len(next_states), p=[lam / composite_exponential.lam for lam in lambdas])
        return {"time": wait_time, "state": next_states[new_state_n]}

    def get_metastate_graph(self):
        """Create a graph of the metastates of the state space. Collect states with the same active tasks."""
        if self.graph == {}:
            raise ValueError("State space graph is empty. Construct it first.")
        if self.states_per_metastate != {}:
            raise ValueError("Metastate graph already constructed.")

        for state in self.graph:
            metastate = MetaState.from_state(state)
            self.states_per_metastate.setdefault(metastate, []).append(state)

            if metastate not in self.metagraph:
                self.metagraph[metastate]: Dict[str, Set[Tuple[int, MetaState]]] = {"s": [], "f": []}
            # don't need to consider progress transitions: same metastate
            for transition_type in ["s","f"]:
                for task_id, next_state in self.graph[state][transition_type]:
                    next_metastate = MetaState.from_state(next_state)
                    next_tuple = (task_id, next_metastate)
                    if next_tuple not in self.metagraph[metastate][transition_type]:
                        self.metagraph[metastate][transition_type].append(next_tuple)

    def visualize_graph(
            self,
            metastate_mode: bool = True
    ) -> None:
        """Create a graph of the graph.

        :param metastate_mode: If True, group states with the same active tasks in the same metastate.
            If False, show all states separately (can be convoluted for large projects).
        """

        if metastate_mode:
            self.get_metastate_graph()
            Atom = MetaState  # the smallest class to group on
            graph = self.metagraph
            transition_types = [t for t in self.transition_types if t != "p"]
            initial_state = MetaState.from_state(self.initial_state)
            final_state = MetaState.from_state(self.final_state)
        else:
            Atom = State  # the smallest class to group on
            graph = self.graph
            transition_types = self.transition_types
            initial_state = self.initial_state
            final_state = self.final_state

        state_ranks: Dict[int, List[Atom]] = {k: [] for k in range(2*len(self.tasks)+1)}

        for state in graph.keys():
            state_ranks[state.rank].append(state)

        # see how many states map to each state in the graph
        inverse_state_graph: Dict[Atom, List[Atom]] = {state: [] for state in graph.keys()}
        for state, transitions in graph.items():
            for _, next_state in transitions["s"] + transitions["f"]:
                inverse_state_graph[next_state].append(state)

        # get position of each state on the canvas:
        state_positions: Dict[Atom, Tuple[int, int]] = {}

        def avg_pos_precursors(_state) -> float:
            if not inverse_state_graph[_state]:
                return 0
            return sum(state_positions[prev][1] for prev in inverse_state_graph[_state]) / len(inverse_state_graph[_state])

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
            return mpatches.Arrow(
                start_pos[0],  # x
                start_pos[1],  # y
                end_pos[0] - start_pos[0],  # dx
                end_pos[1] - start_pos[1],  # dy
                width=0.3,
                alpha=0.5
            )

        # create the line collection data for the transitions
        sf_collections = {t: [] for t in transition_types}  # for the lines of task starting, finishing, progressing
        for state, transitions in graph.items():
            for lab_letter in transition_types:
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

        pc = PatchCollection(
            patches=sum([sf_collections[t] for t in transition_types], []),
            color=["r"]*lsc+["b"]*lsc+([] if metastate_mode else ["g"]*len(sf_collections["p"]))
        )


        if metastate_mode:
            sizes = [str(len(self.states_per_metastate[ms])) for ms in state_positions.keys()]
        else:
            sizes = ["s"]*len(state_positions)

        # plot the states as dots
        #ax.scatter(*zip(*state_positions.values()), s=sizes, color="black")
        for x, y, size in zip(*zip(*state_positions.values()), sizes):
            if x == 0 or x == 2*len(self.tasks):
                continue
            ax.text(x, y, size,
                    bbox={"boxstyle": "circle", "color": "grey"})

        for label,if_state,col in [("Initial", initial_state, "m"), ("Final", final_state, "y")]:
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
        label = "# states" if metastate_mode else "state"
        if not metastate_mode:
            handles.append(mpatches.Arrow(0,0,1,0,color="g", label="progress task"))
        handles.append(mpatches.Circle((0,0),0.1,color="grey", label=label))

        ax.legend(handles=handles)

        # x label
        ax.set_xlabel("Progress in project execution")
        ax.set_ylabel("Number of possible states")
        pref = "Meta" if metastate_mode else ""
        ax.set_title(f"{pref}State Space Transition Graph")

        plt.show()


class MetaState:
    """A collection of states where the same tasks are active, but in different stages.

    :param waiting_states: The task ids that are waiting in this metastate.
    :param active_states: The task ids that are active in this metastate.
    :param finished_states: The task ids that are finished in this metastate.
    """

    def __init__(
            self,
            waiting_states: Iterable[int],
            active_states: Iterable[int],
            finished_states: Iterable[int],
    ):
        self.waiting_states = tuple(sorted(waiting_states))
        self.active_states = tuple(sorted(active_states))
        self.finished_states = tuple(sorted(finished_states))
        self.n_tasks = len(self.waiting_states) + len(self.active_states) + len(self.finished_states)

    def __hash__(self):
        return hash((self.waiting_states, self.active_states, self.finished_states))

    @classmethod
    def from_state(cls, state: State) -> MS:
        """Create a metastate from a state."""
        waiting_states, active_states, finished_states = [], [], []
        for h, stage in enumerate(state):
            if stage == 0:
                waiting_states.append(h)
            elif stage == State.finished:
                finished_states.append(h)
            else:
                active_states.append(h)
        return cls(
            waiting_states=waiting_states,
            active_states=active_states,
            finished_states=finished_states,
        )

    def __eq__(self, other):
        return (
                self.waiting_states == other.waiting_states
                and
                self.active_states == other.active_states
                and
                self.finished_states == other.finished_states
        )

    @property
    def rank(self) -> int:
        """Return the depth of the state in the state space graph: number of vertices traversed to reach it."""
        return 2*len(self.finished_states) + len(self.active_states)


