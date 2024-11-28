# Stochastic Resource-Constrained Project Scheduling Tools

This is a project used to explore, solve, and visualize SRCPSPs, or Stochastic Resource-Constrained Project Scheduling Problems.

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

The idea is to work in terms of the graph below, made with this repo.

![Toy Graph](readme_imgs/project_graph.png)

This framework is particularly powerful when using exponentially distributed random variables, thanks to the memoryless property. This is what we get when we model the rate of probability for a task to finish as constant in time.
For exponentials, and only exponentials, we do not need to keep track of how long a task has been running, which started first, etc. Each state is completely characterized by the set of tasks that are currently Active, Finished, or not yet started (Idle).

In the graph, the state of each task is represented by a letter, and the project states are represented by triplets of letters. Transitions are arrows indicating which task starts or finishes. The term _metastate_, in short, means the tasks may have various stages of completion collected into one node. More in the next section. 

### Erlang Distributed Tasks

With slight admin, this basic framework can be augmented to allow each task to have multiple identical stages, that must finish in sequence. Under the hood, these are separate subtasks that must start upon the completion of the previous stage. However, it is more realistic as it allows
for tasks with an Erlang duration distribution: the Erlang(k,λ) is the distribution of the sum of k exponentially distributed random variables of rate λ. Notably, an Erlang doesn't have its mode at zero. The price is, of course, added computational complexity. 

See an example of the distribution of three tasks, all with λ=5, the first consists of 1 stage, the second of 2 stages, and the last of 5 stages. Changing λ only scales the time axis. N.B. this plot can be produced using the `src/utils.py` function `plot_exponential_vs_erlang`.

![Erlang Graph](readme_imgs/erlang.png)

The Erlang is a special case of the hypoexponential distribution, with all λs equal. Among hypoexponentials with a fixed mean, the Erlang has the smallest variance. Thus Erlang is not only simpler to describe, conversely allowing for different λs per stage is also less desirable from a modeling perspective. I have therefore not implemented the more general hypoexponential distribution.

## Execution Policies

Despite the resource requirements, and dependency requirements per task, there is still superexponential freedom in choosing the scheduling order. The most concise (but less powerful) way to encode our order preference, is with a _policy_.

A policy is simply a permutation of the task indices. I.e. for tasks (0,1,2), a policy could be [2,0,1]. Executing it means always starting the first idle task in the list that is allowed by the constraints. If no task is allowed, we wait until an active one finishes, then return to the policy. It is clear that there are n! policies for n tasks, though in execution they might not all be unique, due to constraints. 

They are also 'blind', not taking into account the history of the project. This is where the next section comes in.

### Contingent Stochastic Dynamic Programming

I have developed and coded a type of contingent Dijkstra's algorithm for the shortest traversal of the state-space graph from Project start to project finish. Let us call this technique Contingent Stochastic Dynamic Programming, or CSDP.
Executing a project in CSDP, we keep a lookup table of which tasks to start (or whether to wait), given that we find ourselves in any project state, such that the expected remaining duration is minimized. The reasoning hinges on linearity of expectations. Even if we are pushed off the expected course, because a task that should have taken long, takes only a short time, or vise versa, we can still use the lookup table to find the optimal next task to start, _contingent_ on inadvertently being in this state.

In the graph first, this is represented by the red arrows. If a state has a red arrow leaving from it, then it is time-efficient to start that task as soon as we arrive at this state. We cannot predict which of the blue arrows will be traversed a priori (i.e. from FAA, we might end up in FFA or FAF), but we can always take a red one when presented.

The astute reader will notice that we cannot perform fully the second phase of traditional Dijkstra's. This is because at many vertices, we don't control which edge we take (which task finishes first). So we can't simply distill the shortest path from the potential landscape we constructed, and throw away the rest. During traversal, we need still need it.

### Obtaining the state-space graph

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

In summary, task dependencies and resource availability determine the possible states, and the possible transitions. State-space is built recursively. The repo implements this in the module `src/StateSpace.py` with the method `StateSpace._graph_from_tasks()`. Once this graph is fully constructed, we can use it to fill out the CSDP contingency table.

### Obtaining the contingency table

This section contains, in my eyes, the most important original contribution of the repo. I will explain obtaining the contingency table with the aid of a tiny toy project. There are two tasks, no constraints, and their durations are exponential / Erlang, tabulated below:

| Task id | # Stages | Avg. time per stage | Dependencies | Resource requirements |
|---------|----------|---------------------|--------------|-----------------------|
| 0       | 2        | 2 units             | -            | 0                     |
| 1       | 1        | 3 units             | -            | 0                     |

The topology of the state-space is graphed below.

![Toy Graph](readme_imgs/CSDP.png)

In this graph:
- The round buttons are states. Their label describes the progress of the two tasks, separated by a pipe (|). 
- The first state can be Idle (0/2), Active (1/2 or 2/2), or Finished (f). The second state can be Idle (0/1), Active (1/1), or Finished (f). 
- Beneath the states is printed the expected time-to-finish, _contingent on taking the optimal path_. 
- E.g. the expected duration of the entire project is t=5.1 units of time, as seen in the initial state in cyan. 
- The arrows between states indicate whether a task is starting (s), progressing (p), or finishing (f), and carry the expected time for that transition to take place _in isolation_.
- Starting a task is immediate.

An important note: When multiple tasks are running, due to the memoryless property, the expected time until the first one of them progresses a stage / finishes (and therefore, the state transitions) is lower than the minimum of expected times for each individual task. The exact formula follows below. 

For each state, we wish to calculate the expected time-to-finish (t) from that state. The contingency table will always point a state to the connecting next state with the lowest expected time-to-finish. 
In the code, the expected time-to-finish is calculated recursively. We call a function `StateSpace.dynamic_step(initial_state)`, which returns t. It, in turn, will query `dynamic_step` on all the states the initial state can evolve into, etc. Thus the algorithm spreads out to all the reachable state space. Once a definite expected-time-to-finish is returned for a state, because all its constituents are known, it is stored using memoization, to avoid duplicate calculations.
However, the easiest way to explain the logic, is the opposite of the way it is coded. We will work our way _up_ the call stack, after the deepest level of recursion has been invoked. 

Namely, for the final state, the expected time-to-finish is trivially 0. See the yellow button in the graph above.

Then, from states that directly evolve into that state (in this case, only through task-finishing transitions), it is also clear that the expected remaining duration is the expected stage duration for the active task. Therefore, the two gray buttons connecting to end state have expected time-to-finish t=2 and t=3, respectively.

Now it gets more interesting. What happens when a state can evolve in multiple directions? In this case, we distinguish two competing transition types: 
1. starting a task: the fastest start option
2. waiting for a task to finish: the wait option.

Let "S;c" be the state S, where task c has been evolved: if it was idle, now it is active, if it was active, it has progressed or finished if it was in its last stage.
From state S, let there be idle tasks (a_1, a_2, ... a_m) that could be started individually, and tasks (b_1, b_2, ... b_n) that are active already. We collect the expected time-to-finish, contingent on starting any of the tasks, or waiting for an (unknown) task to finish. The time-to-finish of state S is the minimum of those options:

    t(S) = min{ t(S| start a_1), t(S| start a_2), ... t(S| start a_m), t(S| wait) }

What is the expected time-to-finish, after starting a task a_1? Because starting a task is immediate, it is simply the expected time-to-finish of the states obtained by starting a_1. The recursion will call that value, and eventually return it.

    t(S| start a_1) = t(S;a_1)

What is the expected time-to-finish, contingent on taking the wait option? If there is only one active task (n=1), it's the expected duration of that (stage of) the task, plus the time-to-finish of the resultant state.

    t(S| wait) = E[task b_1] + t(S;b_1) = 1/λ_b_1 + t(S;b_1)

If there are no active tasks (n=0), t(S| wait) is infinite.

If n>1, the general case, is quite elegant. The λ parameter in an exponential distribution is the rate at which a Poisson process occurs. If multiple Poisson processes are running, the rate at which the first one finishes is the sum of the rates of the individual processes. Thus, the expected time until transition is the average of an exponential distribution with _composite_ λ equal to the sum of the λs of tasks b_1, b_2, ... b_n: 

    E[tasks b_1, b_2, ... b_n] = 1/(λ_b_1 + λ_b_2 + ... + λ_b_n).

After the first task finishes, we must add the expected time to finish of the resultant state. This is the linear probabilistic superposition of the expected times of the possible outcomes (each task could have finished), weighted by their probability. The weights are, of course, commensurate with the rates as well.

    t(S| wait) = (1 + λ_b_1 * t(S;b_1) + λ_b_2 * t(S;b_2) + ... + λ_b_n * t(S;b_n) ) / (λ_b_1 + λ_b_2 + ... + λ_b_n)

So, at each node, the recursive algorithm takes the set of tasks that can start, progress, or finish from the state space (see the previous section), and calls the `dynamic_step` function on each of them. It calculates the composite exponential (if applicable), and set the time-to-finish of the state as the minimum of the options. In the contingency table, we store what option that was.

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


N.B. in the code, whenever a distribution is queried for its expectation, instead we take a quantile, this quantile is set globally for all tasks. Formally, this is, of course, nonsense, because there is no linearity of quantiles property in statistics. But it allows for more freedom to run a heuristic algorithm of the same shape in 'optimistic' or 'worst-case-scenario' planning mode, by setting it to a high or low quantile.
Moreover, for exponential distributions, there is a specific quantile that coincides with the expectation: 1-1/e. It is independent of the rate parameter λ. This is the default value, and by choosing it, we recover exactly the math from above.

Now that the theory is explained: how should one use these tools? On to the next section.

## Getting Started

If you would simply like to play around with the repo out of the box, I suggest you install the `requirements.txt` file, and run `main.py`. It contains examples of the classes and how to interact with them, along with some comments.

The basic flow of this repo is:

1. Create a number of `Task` objects, with dependencies and resource requirements, and put them in a `Project` object.
2. Visualize the `StateSpace` graph of the project, and understand the choices that can be made in execution.
3. Formulate any number of static policies (`Policy`), and see their performance on the project.
4. Use the CSDP to find the dynamic policy, in terms of the contingency table.
5. Compare the performance of the dynamic policy to the static policies in an `Experiment` with a large number of (stochastich) simulations of the project.

## Modules in this Repo

The main modules in `src` are:

- `Project.py`
- `Policy.py`
- `StateSpace.py`

and less fundamental:
- `Experiment.py`,

which all borrow from the `Objects.py` classes:

- `Task`
- `Resource`
- `ProbabilityDistribution`.

Let's start with the objects.

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
    - Generate a task with random resource requirements and duration.
  - Arguments:
    - `task_id`: integer denoting the task ids
    - `dependencies`: list of task ids that this task depends on (must be done before)
    - `max_dependencies`: maximum number of dependencies randomly chosen
    - `resource_types`: number of resource types or list of resource types
    - `max_simultaneous_resources_required`: maximum number of resources required
    - `duration_average_range`: range of average duration
    - `duration_variance_range`: range of variance of duration
    - `prob_type`: type of probability distribution for duration, can be "uniform", "binomial", "random", "exponential", "erlang". For the latter 2, uses an `ExponentialDistribution` object. For the former three, uses an `IntProbabilityDistribution` object with various probability protocols.
    - `max_stages`: maximum number of stages for any task
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

The `Policy`, `DynamicPolicy`, and `Experiment` classes are created to interact with the `Project` class. It in turn leverages the `StateSpace` and `Task` classes.

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
  - Arguments:
    - `metastate_mode`: a boolean, if True, group states into `MetaState` objects, which are sets of states with the same active tasks. If False, show states with tasks in different stages of completion as separate nodes. Very quickly becomes too convoluted.
    - `rich_annotations`: a boolean, if True, show the task ids that start (s), progress (p) or finish (f) between states as annotations on the edges. Also show in the nodes the status of each task of the project. If False, only show the number of states per `MetaState`.
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
    - Create a project from a configuration object. If the configuration is a `RandomConfig`, the tasks will be generated randomly. If it is a `LiteralConfig`, the tasks will be created to the specifications in the configuration.
  - Arguments:
    - `config`: a `Config` object, either a `RandomConfig` or a `LiteralConfig`. For the syntax, there are examples in the `config.py` file, and the `Task` class.
  - Returns:
    - A `Project` object.

The `Project` instance has the following properties:

- `.max_time`: int, the maximum time it would take to complete all tasks in sequence, for the 0.999 quantile of the duration distributions. Useful for setting simulation horizons.
- `.n_topological_orderings`: int, the number of distinct orders in which the tasks can be carried out, considering only dependencies (not resource constraints). A proxy for the complexity of the project.

## Policy.py

This module contains two classes, `Policy` and `DynamicPolicy`, that interact with the `Project` class. They are used to determine the timing of execution of tasks in the project.

### Policy

The `Policy` class implements the classic static functionality of a project scheduling policy. It has the following methods.

- `.__init__()`:  
  - Usage:
    - Initialize a Policy object for a given project.
  - Arguments:
    - `project`: The `Project` instance to which the policy applies.
    - `policy`: The policy, a list of task ids in order of priority. Default is None, then it will be generated according to instructions.
    - `policy_gen`: The method of generating the policy if it is None. Either "random" or "sequential". Default is "random".
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

The method of execution is described in the section _Contingent Stochastic Dynamic Programming_. Starting from the initial state, the `self.project.contingency_table` indicates which tasks to start.
Until the state reaches the final state, the `execute` method always follows the council of the table, starting optimal tasks (with no delay) or waiting when no optimal task can be started.

When waiting, the `StateSpace.get_wait_options` method 
- constructs the composite exponential of all active tasks, 
- samples it for a duration realization, 
- randomly chooses an active task with probability commensurate with its rate.
Then the state progresses to a neighboring state in the graph with that task progressed or finished.

## StateSpace.py

The state-space module contains a large amount of the logic of the repo. Despite the fact that the user may not directly interact with this class, but more through the `Policy` and `Project` classes, it is instructive to understand the methods and properties of the `StateSpace` and `State` classes. 

They are used to model the constituents and transitions of the project execution, following the theory of the section _Statespace Graph Framework..._.

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

### MeteState


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
        """Return the unique lexicographic position of the state inside its state space.

        Initial state is 0, and the final state is the largest one.
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



## Experiment.py

### Experiment

## Sources
```
1: Cremers, S. (2015): Minimizing the expected makespan of a project with stochastic activity durations under resource constraints
2: Kulkarni, V., & Adlakha, V. (1986): Markov and Markov-regenerative PERT networks
```
