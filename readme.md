# Stochastic Resource-Constrained Project Scheduling Tools

This is a project used to explore, solve, and visualize SRCPSPs, or Stochastic Resource-Constrained Project Scheduling Problems.

The main contribution is a Dynamic Programming (DP) implementation of the Markov Decision Process (MDP) of navigating from state to state in the project. 

### Definition

An SRCPSP is a project scheduling problem. It asks: _when to start tasks such that all are completed in the shortest time possible?_ The time between start of the first task and the completion of the last task is called the _makespan_. Some characteristics:
- Each task has a duration that is a random variable. We know its distribution.
- Each task has dependency constraints, meaning that some tasks must finish before others can start. 
- Resources are limited, each task occupies a predefined amount of each resource while active. 

On the last point: the resources are renewable, meaning the total amount of concurrent use is limited, but when a task finishes, the resources it was using become available for other tasks. If task a requires 5 units of resource 'crane', and task b requires 3 units of 'crane', then they can run simultaneously if the total available 'crane' is 8 or more. If not, they must wait for each other.

If we have a project with three tasks, (0,1, and 2), and task 1 has task 0 as a dependency, then their Gantt chart might look like:

```
Task 0:  |-----------------|
Task 1:                    |-----------------|
Task 2:  |------------------------|
----------------> time -> --------------------
```

Unless, for instance, task 0 and task 2 require the same resource, and together they require more than is available. Then we might schedule as follows:

```
Task 0:  |-----------------|
Task 1:                    |-----------------|
Task 2:                    |------------------------|
-------------------> time -> ------------------------
```

increasing the makespan. We could start with task 2, but this would be even slower, because no tasks could run simultaneously.

## Statespace Graph Framework and Memoryless tasks

I borrow heavily from the literature as produced by Cremers (1), who continued the work of Kulkarni and Adlakha (2). The seminal idea is to formulate the state space of the project as a graph, with allowed transitions between states when a particular task starts or finishes. Whether a transition is allowed, is determined by satisfying both the resource constraints and the dependency constraints. 

Let us return to the example project with 3 tasks. Assume we have have 1 unit of one type of 'resource' available, and the following table holds:

| Task id | Dependencies | Resource requirements |
|---------|--------------|-----------------------|
| 0       | -            | 1                     |
| 1       | 0            | 0                     |
| 2       | -            | 1                     |

The idea is to work in terms of the graph below, [made with this repo](#project).

![Toy Graph](readme_imgs/project_graph.png)

This framework is particularly powerful when using exponentially distributed random variables, thanks to the memoryless property. This is what we get when we model the rate of probability for a task to finish as constant in time.
For exponentials, and only exponentials, we do not need to keep track of how long a task has been running, which started first, etc. Each state is completely characterized by the set of tasks that are currently Active, Finished, or not yet started (Idle).

This is an example of a Markov Decision Process (MDP):
sometimes we alter the state with a decision, and sometimes the state is altered stochastically. We cannot predict with certainty which of the active tasks will be completed first.

In the graph, the status of each task is represented by a letter, and the project states are represented by triplets of letters. Transitions are arrows indicating which task starts or finishes. The term _metastate_, in short, means the tasks may have various stages of completion collected into one node. More in the [next subsection](#erlang-distributed-tasks). 

### Erlang Distributed Tasks

With slight admin, this basic framework can be augmented to allow each task to have multiple identical stages, that must finish in sequence. Under the hood, these are separate subtasks that must start upon the completion of the previous stage. However, it is more realistic as it allows
for tasks with an Erlang duration distribution: the Erlang(k,λ) is the distribution of the sum of k exponentially distributed random variables of rate λ. Notably, an Erlang doesn't have its mode at zero. The price is, of course, added computational complexity. 

See an example of the distribution of three tasks, all with λ=5, the first consists of 1 stage, the second of 2 stages, and the last of 5 stages. Changing λ only scales the time axis. N.B. this plot can be produced using the `src/utils.py` function `plot_exponential_vs_erlang`.

![Erlang Graph](readme_imgs/erlang.png)

The Erlang is a special case of the hypoexponential distribution, with all λs equal. Among hypoexponentials with a fixed mean, the Erlang has the smallest variance. Thus Erlang is not only simpler to describe, conversely allowing for different λs per stage is also less desirable from a modeling perspective. I have therefore not implemented the more general hypoexponential distribution.

## Execution Policies

Despite the resource requirements, and dependency requirements per task, there is still often great freedom in choosing the scheduling order. Indeed, without constraints, the freedom is superexponential in the number of tasks. The most concise (but less powerful) way to encode our order preference, is with a _policy_.

A policy is simply a permutation of the task indices. I.e. for tasks (0,1,2), a policy could be [2,0,1]. Executing it means always starting the first idle task in the list that is allowed by the constraints. If no task is allowed, we wait until an active one finishes, then return to the policy. It is clear that there are n! policies for n tasks, though in execution they might not all be unique, due to constraints. 

They are also 'blind', not taking into account the history of the project. This is where the [next subsection](#contingent-stochastic-dynamic-programming) comes in.

### Contingent Stochastic Dynamic Programming

I have implemented a type of contingent Dijkstra's algorithm for the shortest traversal of the state-space graph from project start to project finish. It's an MDP with the makespan as the reward. Let us call this technique Contingent Stochastic Dynamic Programming, or CSDP.

Executing a project in CSDP, we keep a lookup table of which tasks to start (or whether to wait), given that we find ourselves in any project state, such that the expected remaining duration is minimized. The reasoning hinges on linearity of expectations. Even if we are pushed off the expected course, because a task that should have taken long, takes only a short time, or vise versa, we can still use the lookup table to find the optimal next task to start, _contingent_ on inadvertently being in this state.

In the graph first, this is represented by the red arrows. If a state has a red arrow leaving from it, then it is time-efficient to start that task as soon as we arrive at this state. We cannot predict which of the blue arrows will be traversed a priori (i.e. from FAA, we might end up in FFA or FAF), but we can always take a red one when presented.

The astute reader will notice that we cannot perform fully the second phase of traditional Dijkstra's. This is because at many vertices, we don't control which edge we take (which task finishes first). So we can't simply distill the shortest path from the potential landscape we constructed, and throw away the rest. During traversal, we need still need it.

### Obtaining the State-Space Graph

A necessary precursor to navigating the state-space, is mapping it out. A priori, we could imagine any state with any combination of idle, active, and finished tasks. In technical terms, with `n` states, the `n`-ary Cartesian product of the `n` individual task state spaces forms a superset of the project state space. But of course, many of these states are infeasible, and can be discarded. Evidently:
- States are infeasible if the any of the cumulative resources required by active tasks exceed those available.
- States are infeasible if tasks are active or finished, while their dependencies are not finished.

It is not clear whether there is a non-empty set of tasks that may be feasible under these rules, but simply unreachable from the initial state. Nonetheless, our method obtains all reachable states by construction. Run the following pseudocode:

0. Add the initial state to the state space as a vertex. Make it the _current_ state. Keep track of states that have been current already.
1. With the current state, do the following:
   1. Consider which tasks can be started. This is the set of tasks
      - that are idle in the current state, 
      - and whose dependencies are all finished in current state,
      - and whose starting would not exceed the resource constraints given the tasks that are already currently active.
   2. For each of these tasks, add to the state-space the state that differs from the current state by the activation of that task. It's a vertex. Connect it to the current state by a directed (s) edge.
   3. All tasks that are active in the current state can progress or finish.
   4. For each of these tasks, add to the state-space the state that differs from the current state by the progression or finishing of that task. It's a vertex. Connect it to the current state by a directed (p) or (f) edge.
2. For each state found in steps 1.ii and 1.iv, repeat step 1 with it as the current state. If a state was current in an earlier iteration, skip it.
3. When all states added have been current through steps 1.i-1.v once, the state space is complete. Terminate.

In summary, task dependencies and resource availability determine the possible states, and the possible transitions. State-space is built recursively. The repo implements this in the module `src/StateSpace.py` with the method `StateSpace._graph_from_tasks()`. 
Once this graph is fully constructed, we can use it to fill out the CSDP contingency table. Or we can visualize it, see `StateSpace.visualize_state_space()` in the [StateSpace docs](#statespace).

### Obtaining the Contingency Table

This subsection contains, in my eyes, the most important original contribution of the repo. I will explain obtaining the contingency table with the aid of a tiny toy project. There are two tasks, no constraints, and their durations are exponential / Erlang, tabulated below:

| Task id | # Stages | Avg. time per stage | Dependencies | Resource requirements |
|---------|----------|---------------------|--------------|-----------------------|
| 0       | 2        | 2 units             | -            | 0                     |
| 1       | 1        | 3 units             | -            | 0                     |

The topology of the state-space is graphed below.

![Toy Graph](readme_imgs/CSDP.png)

In this graph:
- The round buttons are states. Their label describes the progress of the two tasks, separated by a pipe (|). 
- The first state can be Idle (0/2), Active (1/2 or 2/2), or Finished (f). The second state can be Idle (0/1), Active (1/1), or Finished (f). 
- Beneath the states is printed the expected time-to-finish, _contingent on taking the optimal direction whenever possible_. (Bellman's priciple of optimality)
- E.g. the expected duration of the entire project is t=5.1 units of time, as seen in the initial state in cyan. 
- The arrows between states indicate whether a task is starting (s), progressing (p), or finishing (f), and carry the expected time for that transition to take place _in isolation_.
- Starting a task is immediate.

An important note: When multiple tasks are running, due to the memoryless property, the expected time until the first one of them progresses a stage / finishes (and therefore, the state transitions) is lower than the minimum of expected times for each individual task. The exact formula follows below. 

For each state S, we wish to calculate the expected time-to-finish t(S) from that state, given that all subsequent decisions are optimal. The contingency table will always point a state to the connecting next state with the lowest expected time-to-finish. 
In the code, the expected time-to-finish is calculated recursively. We call a function `StateSpace.dynamic_step(initial_state)`, which returns t. It, in turn, will query `dynamic_step` on all the states the initial state can evolve into, etc. Thus the algorithm spreads out to all the reachable state space. Once a definite expected-time-to-finish is returned for a state, because all its constituents are known, it is stored using memoization, to avoid duplicate calculations.
However, the easiest way to explain the logic, is the opposite of the way it is coded. We will work our way _up_ the call stack, after the deepest level of recursion has been invoked. 

Namely, for the final state, the expected time-to-finish is trivially 0. See the yellow button in the graph above.

Then, from states that directly evolve into that state (in this case, only through task-finishing transitions), it is also clear that the expected remaining duration is the expected stage duration for the active task. Therefore, the two gray buttons connecting to end state have expected time-to-finish t=2 and t=3, respectively.

### Bellman Equation for the Expected Time-to-Finish

Now it gets more interesting. What happens when a state can evolve in multiple directions? In this case, we distinguish two competing transition types: 
1. starting a task: the fastest start option
2. waiting for a task to finish: the wait option.

Let "S;c" be the state S, where task c has been evolved: if it was idle, now it is active, if it was active, it has progressed or finished if it was in its last stage.
From state S, let there be idle tasks (a_1, a_2, ... a_m) that could be started individually, and tasks (b_1, b_2, ... b_n) that are active already. We collect the expected time-to-finish, contingent on starting any of the tasks, or waiting for an (unknown) task to finish. The time-to-finish of state S is the minimum of those options:

    t(S) = min{ t(S| start a_1), t(S| start a_2), ... t(S| start a_m), t(S| wait) }

This is a version of the Bellman equation. It is recursive, as we will see below.

Some practical questions: what is the expected time-to-finish, after starting a task a_1? Because starting a task is immediate, it is simply the expected time-to-finish of the states obtained by starting a_1. The recursion will call that value, and eventually return it.

    t(S| start a_1) = t(S;a_1)

What is the expected time-to-finish, contingent on taking the wait option? If there is only one active task (n=1), it's the expected duration of that (stage of) the task, plus the time-to-finish of the resultant state.

    t(S| wait) = E[task b_1] + t(S;b_1) = 1/λ_b_1 + t(S;b_1)

If there are no active tasks (n=0), t(S| wait) is infinite.

If n>1, the general case, is quite elegant. The λ parameter in an exponential distribution is the rate at which a Poisson process occurs. If multiple Poisson processes are running, the rate at which the first one finishes is the sum of the rates of the individual processes. Thus, the expected time until transition is the average of an exponential distribution with _composite_ λ equal to the sum of the λs of tasks b_1, b_2, ... b_n: 

    E[tasks b_1, b_2, ... b_n] = 1/(λ_b_1 + λ_b_2 + ... + λ_b_n).

After the first task finishes, we must add the expected time to finish of the resultant state. This is the linear probabilistic superposition of the expected times of the possible outcomes (each task could have finished), weighted by their probability. The weights are, of course, commensurate with the rates as well.

    t(S| wait) = (1 + λ_b_1 * t(S;b_1) + λ_b_2 * t(S;b_2) + ... + λ_b_n * t(S;b_n) ) / (λ_b_1 + λ_b_2 + ... + λ_b_n)

So, at each node, the recursive algorithm takes the set of tasks that can start, progress, or finish from the state space (see the [previous subsection](#obtaining-the-state-space-graph)), and calls the `dynamic_step` function on each of them. It calculates the composite exponential (if applicable), and set the time-to-finish of the state as the minimum of the options. In the contingency table, we store what option that was.

This example produces the Contingency table:

| State | t   | optimal transition |
|-------|-----|--------------------|
| ff    | 0.0 | -                  |
| f1    | 3.0 | wait               |
| 2f    | 2.0 | wait               |
| 1f    | 4.0 | wait               |
| 0f    | 4.0 | s0                 |
| 11    | 5.1 | wait               |
| 21    | 3.8 | wait               |
| f0    | 3.0 | s1                 |
| 01    | 5.1 | s0                 |
| 10    | 5.1 | s1                 |
| 20    | 3.8 | s1                 |
 | 00    | 5.1 | s0                 |


N.B. in the code, whenever a distribution is queried for its expectation, instead we take a quantile, this quantile is set globally for all tasks. 

In general, this would nonsense, because there is no linearity of quantiles property in statistics. It simply offers a heuristic algorithm that allows for more 'optimistic' or 'worst-case-schenario' planning.

But specifically for exponential distributions, the following formula holds:

    Quantile(p) = -1/λ * ln(1-p) = - E * ln(1-p)

Meaning the global choice of quantile _is_ linear in the expectation, multiplying the durations by a global factor. It therefore does not change any of the optimal moves in the contingency table, or our strategy. Changing the quantile only scales the expected time-to-finish.

Evidently, there is a specific quantile that coincides with the expectation: 1-1/e. It is independent of the rate parameter λ. This is the default value, and by choosing it, we recover exactly the math from above.

Now that the theory is explained: how should one use these tools? On to the [next section](#getting-started).

## Getting Started

If you would simply like to play around with the repo out of the box, I suggest you install the `requirements.txt` file, and run `main.py`. It contains examples of the classes and how to interact with them, along with some comments.

The basic flow of this repo is:

1. Create a number of `Task` objects, with dependencies and resource requirements, and put them in a `Project` object.
2. Visualize the `StateSpace` graph of the project, and understand the choices that can be made in execution.
3. Formulate any number of static policies (`Policy`), and see their performance on the project.
4. Use the CSDP to find the dynamic policy, in terms of the contingency table.
5. Compare the performance of the dynamic policy to the static policies in an `Experiment` with a large number of (stochastich) simulations of the project.

## Modules in this Repo

The main modules in `src` directory are:

- `Project.py`: [Project docs](#projectpy)
- `Policy.py`: [Policy docs](#policypy)
- `StateSpace.py`: [StateSpace docs](#statespacepy)

and less fundamental:
- `Experiment.py`: [Experiment docs](#experimentpy)

which all borrow from the `Objects.py` classes: 

- `Task`
- `Resource`
- `ProbabilityDistribution`.

The example script `main.py`, and the configuration file `config.py` are in the root directory.

Let's start with `src/Objects.py`.

## Objects.py

### Task

The Task class is the most fundamental object of this repo, of which a project consists. It has the following parameters upon initialization:

- `.__init__()`:
  - Usage:
    - conventional way to initialize a Task object.
  - Arguments:
    - `task_id`: an integer, the unique identifier of the task.
    - `resource_requirements`: a dictionary of `Resource` objects mapping to the integer amount of each required.
    - `duration_distribution`: a `ProbabilityDistribution` object, representing the stochastic duration of the task.
    - `dependencies`: a list of integers, the `task_id`s of the tasks that must finish before this task can start.
    - `stages`: an integer, the number of stages this task consists of. Default is 1.
  - Returns:
    - The initialized Task object.

The `Task` instance has the following methods:

- `.__repr__()`:
  - Usage:
    - What you get when you `print(task)` after `task = Task(...)`.
    - Show all the input params, the distribution, and the minimal dependencies
  - Arguments:
    - None.
  - Returns: 
    - a string representation of the Task object.
- `.duration_realization()`:
  - Usage: 
    - Sample the distribution of the task and return a value with probability according to the distribution.
    - This also progresses the task to the next stage. If there is only one stage, the task is then finished.
  - Arguments:
    - None.
  - Returns:
   - A float or int, depending on the particular implementation of the attached distribution. The value is a realization of the duration distribution.
- `.enough_resources()`:
  - Usage:
    - Check if the task can be executed with the available resources.
  - Arguments:
    - `resources_available`: a dictionary of `Resource` objects mapping to the integer amount of each available.
  - Returns:
    - A boolean, True if the task can be executed with the available resources, False otherwise.
- `.stage_quantile_duration()`:
  - Usage:
    - Return the p-quantile of the distribution of the task.
  - Arguments:
    - `p`: a float, the quantile to return.
  - Returns:
    - A float, the p-quantile of the distribution of the task.

The `Task` instance has the following properties:

- `.average_duration`: float, the average duration of the distribution of the task.
- `.minimal_duration`: int or float, the minimal duration of the distribution of the task. 
- `.maximal_duration`: int or float, the maximal duration of the distribution of the task. If continuous, this is infinity.

There are two class methods, as alternative ways to instantiate a Task object, i.e. `task = Task.generate_random(...)`.

- `.generate_random()`:
  - Usage:
    - Generate a task with 
      - random resource requirements
      - random dependencies, 
      - and random parameters of the probabilistic duration distribution.
    - In order to quickly generate a project with many tasks, interrelations, even when you are not feeling very inspired, or to recon for interesting effects.
  - Arguments:
    - `task_id`: integer denoting the task id of this task.
    - `dependencies`: list of task ids that this task depends on (must be done before). If `None`, dependencies are chosen randomly from lower task ids than the current task.
    - `max_dependencies`: maximum number of dependencies to choose randomly. The actual number of dependencies of this task will be chosen randomly, up to this amount.
    - `resource_types`: number of resource types or list of `Resource` types. If number, will randomly choose from the three types. If list, allocate a requirement for each resource type on the list
    - `max_simultaneous_resources_required`: maximum number of resources required per resource type.
    - `minimum_duration_range`: tuple of 2 ints. Especially for discrete distributions, the range of possible minimum durations, sampled uniformly.
    - `duration_spread_range`: tuple of 2 ints. The range of the difference between the minimum and maximum duration, sampled uniformly.
    - `prob_type`: type of probability distribution for duration, can be "uniform", "binomial", "random", "exponential". For the former three, uses an `IntProbabilityDistribution` object with various probability protocols. For the latter, uses an `ExponentialDistribution` object, with average duration chosen uniformly from `duration_spread_range`, and `minimum_duration_range` is unused.
    - `max_stages`: maximum number of stages for any task. Sampled uniformly between 1 and this number.
  - Returns:
    - A Task object.
- `.from_dict()`:
  - Usage:
    - Create a task from a config dictionary.
  - Arguments:
    - `task_dict`: a dictionary with the following keys: `id`, `resource_requirements`, `distribution`, `dependencies`, `stages`.
  - Returns:
    - A Task object.

### Resource

The Resource class is a simple `Enum` that enumerates the resources that can be required by tasks. It has the following members:
- `Resource.centrifuge`
- `Resource.crane`
- `Resource.drill`

One can easily modify the repo to allow for more resource types, or change the names, but care must be taken to then also modify the `config.py` file.

### ProbabilityDistribution

This is a `ABC` parent class, of which two children are currently implemented that inherit from it, `IntProbabilityDistribution` and `FloatProbabilityDistribution`. The `IntProbabilityDistribution` is used for tasks that have integer durations, and the `ExponentialDistribution`. All distributions have the following methods:

- `.realization()`: 
  - Usage:
    - Sample the distribution and return a realization.
  - Arguments:
    - None.
  - Returns:
    - A float or int, depending on the particular implementation of the distribution. The value is a realization of the distribution.
- `.quantile()`:
  - Usage:
    - Return the p-quantile of the distribution.
  - Arguments:
    - `p`: a float, the quantile to return.
  - Returns:
    - A float or int, depending on the particular implementation of the distribution. The value is the p-quantile of the distribution.

They also have the following properties:
- `.average`: float, the average duration of the distribution.
- `.max`: float or int, the maximal duration (worst case scenario) of the distribution. For continuous distributions, this is infinity or a very high quantile, as set in the `config.py` file.

### IntProbabilityDistribution

This is a child class of `ProbabilityDistribution` that is used for tasks that have integer durations. It is characterized by the values and their discrete finite probabilities.
Most of the repo will not work with this type, because it is not memoryless (as evinced by the method `prob_finish_at()`), but it is still useful for some simulations and as a comparison to the exponential distribution.

- `.__init__()`:
  - Usage:
    - Initialize an IntProbabilityDistribution object.
  - Arguments:
    - `values`: a list of integers, the possible values of the distribution.
    - `probabilities`: a list of floats, the probabilities of the corresponding values.
  - Returns:
    - The initialized IntProbabilityDistribution object.
- `.prob_finish_at()`:
  - Usage:
    - Return the probability of finishing at time, given that it did not finish at any earlier time. If queried on a time-step that is not in the support, return 0.
    - For example, if we have arrived on the last value of its support, and the task has not finished yet, the probability of finishing at this time is 1.
    - Useful for timestep simulations.
  - Arguments:
    - `time`: an integer, the time-step to query. If, for instance `time`=4, we get the probability of finishing at time 4, given that the task has not finished at time 3 or earlier.
  - Returns:
    - A float, the probability of finishing at time `time`.

### ExponentialDistribution

The central distribution of this repo. Used for exponentially distributed tasks, and also Erlang, as it will simply be sampled multiple times.

- `.__init__()`:
  - Usage:
    - Initialize an ExponentialDistribution object.
  - Arguments:
    - `lambda`: a float, the rate parameter of the distribution.
  - Returns:
    - The initialized ExponentialDistribution object.

## Project.py

The `Project` class is the main bookkeeping object of the repo. It collects all the moving parts needed to describe an idealized scheduling project.

The `Policy` ([docs](#policy)), `DynamicPolicy` ([docs](#dynamicpolicy)), and `Experiment` ([docs](#experiment)) classes are created to interact with the `Project` class. It in turn leverages the `StateSpace` ([docs](#statespace)) and `Task` ([docs](#task)) classes.

### Project

An instance of the `Project` class has the following methods.

- `.__init__()`:
  - Usage:
    - Initialize a Project object.
    - Will prune the dependencies of the individual tasks, to achieve the minimal set per task that is equivalent to the input. I.e. if task 2 requires [0,1], but task 1 requires [0], then task 2 need only require [1].
    - Will also immediately construct the `StateSpace` object, and the `contingency_table` needed to execute the CSDP.
  - Arguments:
    - `task_list`: a list of `Task` objects, the tasks of the project.
    - `resource_capacities`: a dictionary with `Resource` objects as keys and integers as values, the capacities of the resources.
    - `decision_quantile`: a float, the quantile to use for decision-making in the CSDP. Default is 1-1/e, which turns the quantile into the expectation for exponential distributions.
  - Returns:
    - The initialized Project object.
- `.visualize_state_space()`:
  - Usage:
    - Visualize and plot the graph structure of the corresponding `StateSpace` object of this project.
    - Is simply a wrapper around the `StateSpace.visualize_state_space()` method. See the [StateSpace docs](#statespace).
  - Arguments:
    - `metastate_mode`: a boolean, if True, group states into `MetaState` objects, which are sets of states with the same active tasks. If False, show states with tasks in different stages of completion as separate nodes. Very quickly becomes too convoluted.
    - `rich_annotations`: a boolean, if True, show the task ids that start (s), progress (p) or finish (f) between states as annotations on the edges. Also show in the nodes the status of each task of the project. If False, only show the number of states per `MetaState`.
    - `add_times`: a boolean, if True, show the expected time-to-finish of each state as an annotation, and expected duration added for each individual transition. Only works if `metastate_mode` is False.
  - Returns:
    - None.
- `__repr__()`:
  - Usage:
    - What you get when you `print(project)` after `project = Project(...)`.
    - Shows a table with the attributes of the tasks, then the available resources, duration statistics, and some information about the state-space complexity.
  - Arguments:
    - None.
  - Returns:
    - a string representation of the Project object.
- `.reset_task_stages()`:
  - Usage:
    - Reset the stages of all tasks to 0. When tasks are queried for duration realizations, they count through stages. When simulating a project repeatedly, these must be reset.
  - Arguments:
    - None.
  - Returns:
    - The same `Project` object with all tasks reset to stage 0.
- `prune_dependencies()`:
  - Usage:
    - A `Task` instance may have an arbitrary set of its own dependencies, in isolation. In the context of other tasks in the project, these may be redundant.
    - This method remove dependencies from tasks that are already dependencies of dependencies. 
    - It modifies the internal parameter `minimal_dependencies` of the `Task` objects in this project.
    - Also fills the `full_dependencies` attributes of the `Task` objects, including all dependencies of dependencies.
  - Arguments:
    - None.
  - Returns:
    - None.
- `.print_contingency_table()`:
  - Usage:
    - Print the contingency table of the project in a concise way.
    - Each row is a set of states at which a particular task should be started.
    - Not all states are represented, only the ones where a task should be started according to the CSDP.
  - Arguments:
    - None.
  - Returns:
    - None.

The class has the class method `from_config()` that allows for easy instantiation of a `Project` object from a `Config` object.

- `.from_config()`:
  - Usage:
    - Create a project from a configuration object. Any config must have:
      - `resource_capacities`: dictionary (keys: name of the resource, values: int), the number of resources available for each type. Can also be an int, then all resources have the same capacity.
    - If the configuration is a `RandomConfig`, the tasks will be generated randomly. Then it must have the following fields:
      - `n_tasks`: integer, the number of tasks to generate.
      - `max_dependencies`: integer, the maximum number of dependencies per task. Dependencies are chosen randomly from lower task_ids, up to this number.
      - `max_simultaneous_resources_required`: integer, the maximum number of resources required per task per resource type.
      - `minimum_duration_range`: tuple of 2 ints, the interval from which to choose the minimum duration of each task uniformly.
      - `duration_spread_range`: tuple of 2 ints, the interval from which to choose the difference between the minimum and maximum duration of each task uniformly.
      - `prob_type`: string, can be "exponential", "binomial", "uniform", or "random". The type of probability distribution for duration.
      - `max_stages`: integer, the maximum number of stages for any task. Choose uniformly from 1 to this number, per task.
    - See also the [docs](#task) on the `Task.generate_random()` method.
    - If it is a `LiteralConfig`, the tasks will be created to the specifications in the configuration. Then the config must have the following fields:
      - tasks: a list of dictionaries, each with the following keys: `id`, `dependencies`, `distribution`, `stages`, `avg_stage_duration`, `resource_requirements`.
      - the `resource_requirements` dictionary must have the resource names as keys, and the required amount as values.
  - Arguments:
    - `config`: a `Config` object, either a `RandomConfig` or a `LiteralConfig`. For the syntax, there are examples in the `config.py` file, and the `Task` class.
  - Returns:
    - A `Project` object.

The `Project` instance has the following properties:

- `.max_makespan`: int, the maximum time it would take to complete all tasks in sequence, for the 0.999 quantile of the duration distributions. Useful for setting simulation horizons.
- `.n_topological_orderings`: int, the number of distinct orders in which the tasks can be carried out, considering only dependencies (not resource constraints). A proxy for the complexity of the project.

## Policy.py

This module contains two classes, `Policy` and `DynamicPolicy`, that interact with the `Project` class ([docs](#project)). They are used to determine the timing of execution of tasks in the project.

### Policy

The `Policy` class implements the classic static functionality of a project scheduling policy. It has the following methods.

- `.__init__()`:  
  - Usage:
    - Initialize a Policy object for a given project.
  - Arguments:
    - `project`: The `Project` instance to which the policy applies.
    - `policy`: The policy, a list of task ids in order of priority. Default is None, then it will be generated according to instructions.
    - `policy_gen`: The method of generating the policy if it is None. Either "random": uniformly choose a permutation, or "sequential": in order of task id. Default is "random".
  - Returns:
    - The initialized Policy object.
- `.execute()`:  
  - Usage:
    - Execute the policy until all tasks are completed. Will change the state of all `Task` objects to finished.
  - Arguments:
    - None.
  - Returns:
    - The makespan, or total time taken to complete all tasks.
- `get_gant_str()`:  
  - Usage:
    - Return a string representation of the Gantt chart of the policy
    - Will only work after execution.
  - Arguments:
    - `n_times`: The number of time steps to display in the Gantt chart. Default is 100.
  - Returns:
    - A string representation of the Gantt chart.
- `get_resource_chart()`:  
  - Usage:
    - Return a string representation of the usage of each resource over time.
    - Will only work after execution.
  - Arguments:
    - `n_times`: The number of time steps to display in the resource chart. Default is 100.
  - Returns:
    - A string representation of the resource chart.
- `__repr__()`:  
  - Usage:
    - What you get when you `print(policy)` after `policy = Policy(...)`.
    - Shows the project, the policy, the Gantt chart, and resource usage chart.
    - If called before execution, only the policy is shown.
  - Arguments:
    - None.
  - Returns:
    - a string representation of the Policy object.

While these are the functions intended for user interaction, under the hood, it is interesting to explain two more methods, both called during execution.

- `.evolve()`:  
  - Usage:
    - Execute one time step of the policy.
      1. Every time step, moves forward in time to the next task that will finish or progress: the lowest new value in `future_task_progress`
      2. Frees up the resources that task was occupying.
      3. Updates the `state_sequence` with the new state.
      4. Checks the policy for the next allowed task to start, and adds it to the `docket`. 
      5. Removes the available resources from 
      6. Queries their distributions for a realization, and schedules their completion in `future_task_progress`.
  - Arguments:
    - None.
  - Returns:
    - None.
- `.choose_task_id()`:  
  - Usage:
    - Choose the highest ranked task of the policy that can be executed from the `remaining_policy` list.
    - Considers resource availability and dependencies.
    - removes the task from the `remaining_policy` list.
  - Arguments:
    - None.
  - Returns:
    - The task id of the chosen task, or None if no task can be executed. In the second case, the policy will wait until a task finishes.

### DynamicPolicy

The `DynamicPolicy` class is intended to carry out the CSDP in a simulation. Most of the logic and functionality it uses is, for reasons of convenience, housed in the `Project` class. Most notably, the contingency table. It has two methods:

- `.__init__()`:  
  - Usage:
    - Initialize a DynamicPolicy object for a given project.
  - Arguments:
    - `project`: The `Project` instance to which the policy applies.
  - Returns:
    - The initialized DynamicPolicy object.
- `.execute()`:  
  - Usage:
    - Execute the policy until all tasks are completed. Will change the state of all `Task` objects to finished.
  - Arguments:
    - None.
  - Returns:
    - The makespan, or total time taken to complete all tasks.

The method of execution is described in the section [Contingent Stochastic Dynamic Programming](#contingent-stochastic-dynamic-programming). Starting from the initial state, the `self.project.contingency_table` indicates which tasks to start.
Until the state reaches the final state, the `execute` method always follows the council of the table, starting optimal tasks (with no delay) or waiting when no optimal task can be started.

When waiting, the `StateSpace.wait_for_finish` method 
- constructs the composite exponential of all active tasks, 
- samples it for a duration realization, 
- randomly chooses an active task with probability commensurate with its rate.
Then the state progresses to a neighboring state in the graph with that task progressed or finished.

## StateSpace.py

The state-space module contains a large amount of the logic of the repo. Despite the fact that the user may not directly interact with this class, but more through the `Policy` ([docs](#policy)) and `Project` ([docs](#project)) classes, it is instructive to understand the methods and properties of the `StateSpace` and `State` classes. 

They are used to model the constituents and transitions of the project execution, following the theory of the section on the [Statespace Graph Framework](#statespace-graph-framework-and-memoryless-tasks).

### State

The `State` class is the fundamental building block of the `StateSpace` class. It represents a state of the project. It holds information about each task, how many stages they allow, and in which stage they are currently. It has the following methods:

- `.__init__()`:  
  - Usage:
    - Initialize a State object.
  - Arguments:
    - `total_stages`: a list of integers, the total number of stages of each task, ordered by task id.
    - `current_stages`: a list of integers or strings, the current stage of each task. If a task is finished, the string "f" is used.
    - `error_check`: a boolean, if True, check if the input is valid. If False, skip the check. Default is True.
  - Returns:
    - The initialized State object.
- `.progress_task()`:  
  - Usage:
    - If the task is idle, activate it
    - If the task is active, progress to the next stage, or finish if it was in the last stage.
    - If the task is finished, raise an error.
  - Arguments:
    - `task_id`: an integer, the id of the task to progress.
  - Returns:
    - A new State object, identical, except with the task transitioned.
- `.copy()`:  
  - Usage:
    - Return a copy of the state.
  - Arguments:
    - None.
  - Returns:
    - A new State object, identical to the original.
- `.task_complete()`:  
  - Usage:
    - Check if a task is finished.
  - Arguments:
    - `index`: an integer, the id of the task to check.
  - Returns:
    - A boolean, True if the task is finished, False otherwise.
- `.dependencies_finished()`:  
  - Usage:
    - Check if all dependencies of a task are finished.
  - Arguments:
    - `task`: a `Task` object, the task to check.
  - Returns:
    - A boolean, True if all dependencies are finished, False otherwise.
- `.resources_used()`:  
  - Usage:
    - Return the cumulative resources used by the state.
  - Arguments:
    - `task_list`: The list of `Task` objects, all tasks of the project.
  - Returns:
    - A dictionary with `Resource` objects as keys and integers as values, the resources occupied by the state.

A `State` object has the following properties:

- `.is_initial`: boolean, True if the state is the initial state of the project.
- `.is_final`: boolean, True if the state is the final state of the project.
- `.lexicographic_position`: int, a unique identifier of the state in the state space. Uses a mixed radix number system.
- `.rank`: int, the depth of the state in the state space graph: number of vertices traversed (horizontally) to reach it.
The rank of the state follows the encoded in the class method `rank_from_stage()`. 


    rank(S) = 2 * (# of finished tasks) + (# of active tasks)

The class has some useful dunder methods implemented:

- `.__iter__()`: iterate over the current stages of the state. Enables unpacking, i.e. `for stage in state`.
- `.__getitem__()`: get the current stage of a task by index. i.e. `state[task_id]`.
- `.__len__()`: get the number of tasks in the state, is `len(state)`.
- `.__hash__()`: get the hash of the state. Composed of the hash of the current stages. Necessary to be able to use states as dictionary keys.
- `.__eq__()`: check if two states are equal. Compares the current stages and total stages.
- `.__repr__()`: get a string representation of the state for printing, in brackets. E.g. `<0/1|f|2/2>` for a 3-task state.
- `.__lt__()`: Lesser than operator. Key is the `lexicographic_position`. Necessary for sorting states.
- `.__gt__()`: Greater than operator. Key is the `lexicographic_position`. Necessary for sorting states.

This comparison convention tends to call states 'greater', that has the task with the highest id in the most advanced stage. This is used as a tie breaker in the sorting of the columns of the state space graphs.

### StateSpace

This class is the engine of the repo. It constructs and holds the topology of the project state space graph, and the methods to traverse it. It has the following methods:

- `.__init__()`:  
  - Usage:
    - Initialize a StateSpace object for a given project.
    - Construct the state space graph, which is a dictionary of states, each with a dictionary of possible transitions, in `._graph_from_tasks()` See also [Obtaining the State-Space Graph](#obtaining-the-state-space-graph).
  - Arguments:
    - `tasks`: an ordered list of `Task` objects, the tasks of the project.
    - `resource_capacities`: a dictionary with `Resource` objects as keys and integers as values, the capacities of the resources.
  - Returns:
    - The initialized StateSpace object.
- `.construct_contingency_table()`:  
  - Usage:
    - Construct the contingency table of the project. See also the subsection [Obtaining the Contingency Table](#obtaining-the-contingency-table).
    - This table informs the `DynamicPolicy` class on which tasks to start from which states, or to wait.
    - This method is called by the initialization of the `Project` class, so the contingency table can be stored as a property of the project.
  - Arguments:
    - `decision_quantile`: a float, the quantile to use for decision-making in the CSDP. Default is 1-1/e, which turns the quantile into the expectation for exponential distributions. For the median, choose 0.5.
  - Returns:
    - The contingency table, a dictionary with `State` instances as keys, and integers (task ids) as values.
- `.visualize_state_space()`:  
  - Usage:
    - Visualize and plot the graph structure of the state space. Example result similar to that in [the introduction](#statespace-graph-framework-and-memoryless-tasks).
    - The red arrows convey the information in the contingency table, and point to optimal tasks to start, if present.
    - If `metastate_mode` is True, will also construct the metastate graph using the method `.construct_metastate_graph()`, where states connected by `progress` edges are grouped together.
  - Arguments:
    - `metastate_mode`: a boolean, if True, group states into `MetaState` objects, which are sets of states with the same active tasks. If False, show states with tasks in different stages of completion as separate nodes. Only for small projects should this be False.
    - `rich_annotations`: a boolean, if True, show the task ids that start (s), progress (p) or finish (f) between states as annotations on the edges. Also show in the nodes the status of each task of the project. If False, only show the number of states per `MetaState`.
    - `add_times`: a boolean, if True, show the expected time-to-finish under each state, and expected duration added to the annotation for each individual transition. Only works if `metastate_mode` is False.
  - Returns:
    - None.

In the graph, the horizontal progress counts the number of tasks started plus the number of tasks finished. In other words, the horizontal position is the rank of the state. See the [State docs](#state). For n states, there will be 2*n+1 columns, until all states are finished.
The vertical placement of a state has no formal meaning: they are arranged by a heuristic in order to make the connecting arrows as horizontal as possible, and as a tie-breaker according to their `lexicographic_position`. However, the maximum height does give a sense of the number of simultaneous options for the state of the project at this point in the schedule, and is therefore a proxy for complexity.

In `metastate_mode`, much information is concealed, specifically about the CSDP. It is possible that the contingent optimal task to start is different for two or more states inside the same metastate. Or, from some states, waiting is best, while from others, a task must start.
In `metastate_mode`, that cannot be distinguished visually. A red arrow may thus not be applicable to all states implied in the metastate, or multiple red arrows may point from the same metastate, which is misleading: the actual optimal one depends on the state. Metastates are a convenient shorthand to condense information visually, but for simulations, they are not as accurate.

For the technically inclined reader, I'll go over some internal methods of the `StateSpace` class.

- `.dynamic_step()`:  
  - Usage:
    - Execute one step of the CSDP algorithm. See also the subsection [Obtaining the Contingency Table](#obtaining-the-contingency-table).
    - Calculate the expected time-to-finish of the state, contingent on starting a task, or waiting.
    - Memoization: update the `remaining_path_lengths` dictionary with the expected time-to-finish of the state.
    - Calls itself recursively on all states the input state can transition to. If memoized, escapes immediately.
    - Update the `contingency_table` with the optimal transition if definite.
  - Arguments:
    - `state`: a `State` object, the state for which to calculate the expected time-to-finish.
  - Returns:
    - the time-to-finish t of the state, contingent on the optimal decision.
- `._graph_from_tasks()`:  
  - Usage:
    - Construct the state space graph from the tasks using recursion. See also [Obtaining the State-Space Graph](#obtaining-the-state-space-graph).
    - For each new state, get its descendants. They are divided into sets, depending on the type of transition: start, progress, or finish.
    - Each set is described by a list of new states, and the task that changed status to reach them.
    - E.g. `dict(state1= {"s": [(task_id3, state3), ...], "p": [(task_id4, state4), ...], "f": [(task_id5, state5), ...]}, state2=...)`
    - Main internal function is `._get_descendants()`.
  - Arguments:
    - None.
  - Returns:
    - A nested dictionary describing the state space.
- `._get_descendants()`:  
  - Usage:
    - Return the possible transitions from a state, both due to starting, progressing, and finishing tasks.
    - A transition is possible if the status of exactly one task is different, going from idle to active, progressing to the next stage, or from active to finished.
    - Moreover, a task can only start if all its dependencies are finished, and if there are enough resources available for the task along with all other active tasks.
    - Active tasks can always progress, and can thus finish if in their final stage.
  - Arguments:
    - `state`: a `State` object, the state from which to find the possible transitions.
  - Returns:
    - A dictionary with three keys, "s", "p" and "f", each with a list of tuples.
    - The first element of each tuple is the task id that changes status/stage.
    - The second element is the state that results from the transition.
- `._resources_available()`:  
  - Usage:
    - Return the resources available, i.e. the difference between the resource capacities and the total resources used by the state.
  - Arguments:
    - `state`: a `State` object, the state for which to calculate the available resources.
  - Returns:
    - A dictionary with `Resource` objects as keys and integers as values, the unused resources available.
- `.wait_for_finish()`:  
  - Usage:
    - Simulate waiting for a task to finish and return the time and the state that results from it.
    - This is a helper function for the `DynamicPolicy` class.
  - Arguments:
    - `state`: a `State` object, the state for which to calculate the possible transitions. Must have at least one active task.
  - Returns:
    - A dictionary with two keys, "time" and "state". As values a float: the realization, and a `State` object, the resulting state, respectively.
- `.construct_metastate_graph()`:  
  - Usage:
    - Construct a graph of the metastates of the state space. Collect into the same node states with the same active tasks.
    - Maps each `State` to its `MetaState`, and also the states it transitions to. If they are distinct `MetaState`s, this transition is added to the metastate graph.
    - The graph is stored as `self.metagraph` and has a similar signature to the `self.graph` attribute, but with `MetaState` objects as outers keys, and no progress transitions.
    - Also constructs the `meta_contingency_table` attribute, which is a dictionary with `MetaState` objects as keys, and lists of task ids as values. 
    - This is a helper function for the `visualize_state_space` method.
  - Arguments:
    - None.
  - Returns:
    - None.

### MetaState

The metastate is an auxiliary class used for visual bookkeeping. Formally, a metastate is a set of states that have the same active tasks. I.e. when a task progresses to another stage, but does not finish, the obtained state is still in the same metastate as the old state.

Its main function is to reduce the number of nodes visible in the state space graph. See the [docs above](#statespace) for more information on the metastate mode.

Perhaps it is also helpful abstraction to reduce the apparent complexity of a project, if one thinks of the division of a task into stages as merely an artifact used to obtain the correct probability distribution, not a measure of progress that could be observed realistically.
The `MetaState` class has the following methods:

- `.__init__()`:  
  - Usage:
    - Initialize a MetaState object.
  - Arguments:
    - `idle_states`: a list of integers, the task ids that are waiting to start in this metastate.
    - `active_states`: a list of integers, the task ids that are active in one of their stages in this metastate.
    - `finished_states`: a list of integers, the task ids that are finished in this metastate.
  - Returns:
    - The initialized MetaState object.

The more common way to instantiate a `MetaState` object is through the class method:

- `.from_state()`:  
  - Usage:
    - Create the corresponding `MetaState` object to a `State` object.
  - Arguments:
    - `state`: a `State` instance, the state from which to create the metastate.
  - Returns:
    - A `MetaState` object that contains the state in the input.

The `MetaState` has the following dunder methods implemented:

- `.__hash__()`: get the hash of the metastate. Composed of the hash of the idle, active, and finished states. Necessary to be able to use metastates as dictionary keys.
- `.__eq__()`: check if two metastates are equal. Compares the idle, active, and finished states.
- `.__repr__()`: get a string representation of the `MetaState` for printing, in angle brackets. Each task is described by a letter, "I" for idle, "A" for active, and "F" for finished. E.g. `<FAII>` for a 4-task metastate.

Finally, the class has the following properties:

- `.rank`: int, the depth of the state in the state space graph: number of vertices traversed to reach it. It is 2*(# of finished tasks) + (# of active tasks).
- `.lexicographic_position`: int, a unique identifier of the state in the state space. Isomorphic to the trinary number system, as each task forms a trit.

## Experiment.py

This module stands apart from the others, as it is slightly ad-hoc. It is intended more as an example of how the more fundamental classes can be used for statistical analysis, than as a source of immutable building blocks. 

### Experiment

The `Experiment` class is used to compare the performance of the CSDP to different static policies on a project. It can only be instantiated through a `Config` object, and has the following methods:

- `.__init__()`:  
  - Usage:
    - Initialize an Experiment object.
    - As a component, will initialize an internal `Project` object.
  - Arguments:
    - `experiment_config`: a `Config` object, the configuration of the experiment. Must have the following fields:
      - `n_permutations`: an integer, the number of different policies to compare to the CSDP.
      - `n_runs`: an integer, the number of times to run each policy.
      - `p_value_threshold`: a float between [0,0.5], the threshold for the p-value to consider the CSDP significantly faster than a static policy.
      - All [requirements](#project) of the `Project.from_config()` method.
  - Returns:
    - The initialized Experiment object.
- `.run()`:
    - Usage:
      - Create `n_permutations` static policies, each with a random permutation of the tasks as precedence order.
      - Execute each random `Policy` for `n_runs` times, storing the makespans in `self.results_dict`.
      - Also execute the CSDP `DynamicPolicy` for `n_runs` times, storing the makespans in `self.results_dict`.
      - Policies are either labeled in terms of the precedence order of tasks, or as "CSDP".
      - The execution is probabilistic, by sampling the task distributions, so the results will vary between runs.
    - Arguments:
        - None.
    - Returns:
        - None.
- `.analyze()`:
    - Usage:
      - Print the results of the experiment and show a histogram of the makespans.
      - This histogram also shows the average makespan of each policy.
      - Prints the details of the project, using the `Project.__repr__()`, see its [docs](#project).
      - Prints a report on the significance of the hypothesis that the CSDP is faster than each static policy.
      - a p-value is obtained with the (strong) approximation that the makespans are normally distributed.
    - Arguments:
        - None.
    - Returns:
        - None.

## Sources
```
1: Cremers, S. (2015): Minimizing the expected makespan of a project with stochastic activity durations under resource constraints
2: Kulkarni, V., & Adlakha, V. (1986): Markov and Markov-regenerative PERT networks
```
