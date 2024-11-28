"""Hold the possible states in the state space of the project."""
from typing import List, Dict, Tuple, Union, TypeVar, Iterable, Set

import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

from src.Objects import Task, Resource, ExponentialDistribution
from src.utils import ArrowCoordMaker, HandlerEllipse, HandlerArrow

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
        finished: 2,
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

        # initialize the lexicographic position of the state, as a unique identifier inside its state space
        # will use a mixed radix number system with the total stages (+1) as the radix
        self._lexicographic_position = None

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
        return State(
            total_stages=self.total_stages,
            current_stages=new_stati,
            error_check=False
        )

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
        extremes = [self.finished]
        str_rep = [str(c) if c in extremes else f"{c}/{t}" for c, t in zip(self.current_stages, self.total_stages)]
        return "<"+"|".join(str_rep)+">"

    def __lt__(self, other):
        return self.lexicographic_position < other.lexicographic_position

    def __gt__(self, other):
        return self.lexicographic_position > other.lexicographic_position

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

    @property
    def lexicographic_position(self) -> int:
        """Set and return the unique lexicographic position of the state inside its state space.

        Initial state in the state-space is 0, and the final state is the largest one.

        mixed radix number system:
        # per task a digit, radix is #stages+1
        """
        if self._lexicographic_position is None:
            self._lexicographic_position = 0
            running_digit_size = 1
            for current_stage, max_stage in zip(
                    self.current_stages,
                    self.total_stages):
                current_digit = max_stage + 1 if current_stage == self.finished else current_stage
                self._lexicographic_position += current_digit * running_digit_size
                running_digit_size *= max_stage + 1
        return self._lexicographic_position


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

        # initialize some graph structures
        self.contingency_table: Dict[State, Union[int, None]] = {}
        self.metagraph: Dict[MetaState, Dict[str, List[Tuple[int, MetaState]]]] = {}
        self.states_per_metastate: Dict[MetaState, List[State]] = {}
        self.meta_contingency_table: Dict[MetaState, List[int]] = {}

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

    def get_wait_options(
            self,
            state: State
    ) -> Tuple[List[State],List[float], ExponentialDistribution]:
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
                self.metagraph[metastate]: Dict[str, Set[Tuple[int, MetaState]]] = {
                    self.start: [],
                    self.finish: []
                }
            # don't need to consider progress transitions: same metastate
            for transition_type in [self.start, self.finish]:
                for task_id, next_state in self.graph[state][transition_type]:
                    next_metastate = MetaState.from_state(next_state)
                    next_tuple = (task_id, next_metastate)
                    if next_tuple not in self.metagraph[metastate][transition_type]:
                        self.metagraph[metastate][transition_type].append(next_tuple)

        # now convert the contingency table:
        for state, task_id in self.contingency_table.items():
            local_contingents = self.meta_contingency_table.setdefault(MetaState.from_state(state), [])
            if task_id is None:
                continue
            local_contingents.append(task_id)

    def visualize_graph(
            self,
            metastate_mode: bool = True,
            rich_annotations: bool = False,
            add_times: bool = True
    ) -> None:
        """Create a graph of the state space, with states as vertices and transitions as edges.

        The states are ordered from left to right by the progress of project completion. Whenever a
        task is started or finished, the corresponding state moves one column to the right. There are then
        2*N+1 columns for N tasks. The initial state is on the left, the final state on the right.

        The states are stacked vertically in a column, so the height is a heuristic for the number of different
        paths (choices) that can be taken at that point in the project.

        Arrows indicate transitions, and they cannot point backwards, or form loops. Red for starting a task,
        blue for finishing a task.

        :param metastate_mode: If True, group states with the same active tasks in the same metastate.
            If False, a single task in different stages implies separately plotted states
            (can be convoluted for even medium-sized projects).
            If False, green arrows connect states that differ by a single tasks progressing through its stages.
        :param rich_annotations: If True, annotate arrows with which task is changing status, and fills the states with
            a flag of idle, active, or finished tasks, in metastate mode. Inadviseable for large projects.
            if metastate_mode is False, the stage of each task is shown.
            if False, the number of states in the metastate is printed, or if metastate_mode is False, simply 's'.
        """

        if metastate_mode:
            self.get_metastate_graph()  # construct the graph with vertices collections of similar states
            Atom = MetaState  # the smallest class to group on
            graph = self.metagraph
            transition_types = [t for t in self.transition_types if t != self.progress]
            def optimal_contingent(state: MetaState, task_id: int) -> bool:
                return task_id in self.meta_contingency_table[state]
        else:
            Atom = State  # the smallest class to group on
            graph = self.graph
            transition_types = self.transition_types
            def optimal_contingent(state: State, task_id: int) -> bool:
                return task_id == self.contingency_table[state]

        state_ranks: Dict[int, List[Atom]] = {k: [] for k in range(2*len(self.tasks)+1)}  # initialize the ranks

        for state in graph.keys():
            state_ranks[state.rank].append(state)

        # We want to reduce the presence of arrows pointing steeply up and down, by choosing a smart
        # layout of the states. We will sort the states in a column by the average height of the states
        # in the previous column that point to them.
        # This allows for a single pass heuristic that will hopefully return a reasonable vertical ordering.
        # For this, we must first see how many states map to each state in the graph
        inverse_state_graph: Dict[Atom, List[Atom]] = {state: [] for state in graph.keys()}
        for state, transitions in graph.items():
            for _, next_state in transitions[self.start] + transitions[self.finish]:
                inverse_state_graph[next_state].append(state)

        # get position of each state on the canvas:
        state_positions: Dict[Atom, Tuple[int, int]] = {}

        def avg_pos_precursors(_state) -> Tuple[float, int]:
            isg = inverse_state_graph[_state]
            if not isg:
                return 0, _state.lexicographic_position
            av_pos = sum(state_positions[prev][1] for prev in isg) / len(isg)
            return av_pos, _state.lexicographic_position

        for rank in state_ranks.keys():
            # heuristic: sort by the average height of the precursors
            # hopefully will make arrows as horizontal as possible
            states = sorted(
                state_ranks[rank],
                key=avg_pos_precursors
            )
            for i, state in enumerate(states):
                state_positions[state] = (rank, i+1)

        transition_annotations: Dict[Tuple[float, float], List[str]] = {}  # initialize the annotations

        # smart arrow coord maker, that doesn't allow vertical arrows to overlap.
        arrow_maker = ArrowCoordMaker()

        # construct the arrows and their annotations
        sf_collections = {t: [] for t in transition_types}  # for the lines of task starting, finishing, progressing
        contingent_starts = []  # position of arrows for the optimal starts per state
        for state, transitions in graph.items():
            for lab_letter in transition_types:
                for task_id, next_state in transitions[lab_letter]:
                    arrow, (text_x,text_y) = arrow_maker.make(state_positions[state], state_positions[next_state])
                    sf_collections[lab_letter].append(arrow)
                    arrow_annotation = lab_letter+str(task_id)
                    if add_times and not metastate_mode:
                        if lab_letter == self.start:
                            arrow_annotation += " (t+0)"
                        else:
                            stage_duration = self.tasks[task_id].duration_distribution.quantile(self.decision_quantile)
                            arrow_annotation += f" (t+{str(round(stage_duration,1))})"
                    transition_annotations.setdefault((text_x, text_y),[]).append(arrow_annotation)
                    # check if arrow is contingency table choice, and if so, give it a different color.
                    if lab_letter == self.start and optimal_contingent(state, task_id):
                        contingent_starts.append(len(sf_collections[lab_letter])-1)

        fig, ax = plt.subplots()
        ax.set_xlim(-1, 2*len(self.tasks)+1)
        max_height = max(v[1] for v in state_positions.values())+1
        ax.set_ylim(0, max_height)

        assert len(sf_collections[self.start]) == len(sf_collections[self.finish]), "# start == # finish transitions"

        # draw the arrows
        for t, col in zip(transition_types, ["m", "b", "g"]):
            for h,arrow in enumerate(sf_collections[t]):
                newcol = "r" if (t == self.start and h in contingent_starts) else col
                ax.arrow(
                    *arrow,
                    head_width=0.1,
                    head_length=0.1,
                    ec=newcol,
                    fc=newcol,
                    lw=5,
                    length_includes_head=True,
                )

        def suf_maker(x: Atom):  # suffix maker for the state buttons
            if metastate_mode:
                return str(len(self.states_per_metastate[x]))
            return self.start

        if rich_annotations:

            # If all metastates have exactly one constituent state (probably due to project config set without allowing
            # multiple stages per task), don't clutter annotations by showing the number of states in each metastate.
            variable_state_counts = not all([suf_maker(x) == "1" for x in state_positions.keys()])

            def string_maker(x):
                base_string = str(x)[1:-1]
                if metastate_mode and variable_state_counts:
                    return " "+base_string + f" \n({(suf_maker(x))})"
                elif add_times and not metastate_mode:
                    return base_string + f" \nt={str(round(self.remaining_path_lengths[x],1))}"
                return base_string

            bbox = dict(boxstyle="round", fc="0.8")
            state_button_contents = map(string_maker, state_positions.keys())
            for pos, label in transition_annotations.items():
                ax.annotate("\n".join(label), pos, bbox=bbox, fontsize=12,
                            horizontalalignment='center',
                            verticalalignment='center')
        else:
            state_button_contents = map(suf_maker, state_positions.keys())

        col_dict = {0: "cyan", 2*len(self.tasks): "yellow"}  # first and last column are initial and final states

        # plot the states as dots
        for x, y, button_content in zip(*zip(*state_positions.values()), state_button_contents):
            ax.text(x, y, button_content,
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox={
                        "boxstyle": "circle",  # can be "ellipse" for more vertical space efficiency
                        "facecolor": col_dict.get(x, "grey"),
                        "edgecolor": "black"
                    }
                    )

        if metastate_mode:
            if rich_annotations:
                label = "metastate" + (" (# states)" if variable_state_counts else "")
            else:
                label = "# states"
        else:
            label = "state"

        # create manual symbols for legend
        legend_handles = [
            mpatches.Circle((0,0), 0.1, facecolor="grey", edgecolor="black"),
            mpatches.Circle((0,0), 0.1, facecolor="cyan", edgecolor="black"),
            mpatches.Circle((0,0), 0.1, facecolor="yellow", edgecolor="black"),
            mpatches.Arrow(0,0,1,0, color="r"),
            mpatches.Arrow(0,0,1,0, color="m"),
            mpatches.Arrow(0,0,1,0, color="b"),
        ]
        legend_lables = [label,
                         "initial state",
                         "final state",
                         "optimal start task",
                         "alternative start task",
                         "finish task"]

        if not metastate_mode:
            legend_handles.append(mpatches.Arrow(0,0,1,0, color="g"))
            legend_lables.append("progress task")

        ax.legend(legend_handles,
                  legend_lables,
                  handler_map={
                      mpatches.Circle: HandlerEllipse(),
                      mpatches.Arrow: HandlerArrow()
                  })

        # x label
        ax.set_xlabel("Progress in project execution")
        ax.set_ylabel("Number of possible states")
        # set x and y ticks to only possible values
        ax.set_yticks(range(1,max_height))
        ax.set_xticks(range(2*len(self.tasks)+1))
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

        # mapping from the state to a unique integer according to a trinary system (trit: 0,1,2)
        self.lexicographic_position: int = sum(3**i for i in active_states) + 2 * sum(3**i for i in finished_states)

    def __hash__(self):
        return hash((self.waiting_states, self.active_states, self.finished_states))

    def __repr__(self):
        string_list = [""]*self.n_tasks
        for _symbol, _states in zip("IAF", [self.waiting_states, self.active_states, self.finished_states]):
            for state in _states:
                string_list[state] = _symbol
        return "<"+"".join(string_list)+">"

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


