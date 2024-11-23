# Stochastic Resource-Constrained Project Scheduling Tools

This is a project used to explore, solve, and visualize SRCPSPs, or Stochastic Resource-Constrained Project Scheduling Problems.

### Definition

An SRCPSP is a project scheduling problem. It asks: _when to start tasks such that all are completed in the shortest time possible?_ The time between start of the first task and the completion of the last task is called the _makespan_. Some characteristics:
- Each task has a duration that is a random variable. We know its distribution.
- Each task has dependency constraints, meaning that some tasks must finish before others can start. 
- Resources are limited, each task occupies a predefined amount of each resource while active. 

On the last point: the resources are renewable, meaning the total amount of concurrent use is limited, but when a task finishes, the resources it was using become available for other tasks. If task a requires 5 units of resource 'drill', and task b requires 3 units of 'drill', then they can run simultaneously if the total available 'drill' is 8 or more. If not, they must wait for each other.

## Statespace Graph Framework and Memoryless tasks

I borrow heavily from the literature as produced by Cremers (1), who continued the work of Kulkarni and Adlakha (2). The seminal idea is to formulate the state space of the project as a graph, with allowed transitions between states when a particular task starts or finishes. Whether a transition is allowed, is determined by satisfying both the resource constraints and the dependency constraints.

See the graph below for a toy project with 3 tasks (0,1 and 2). Task 1 has task 0 as a prerequisite. There are no active resource constraints. The graph is made using this repo.

![Toy Graph](readme_imgs/project_graph.png)

This framework is particularly powerful when using exponentially distributed random variables, thanks to the memoryless property. This is what we get when we model the rate of probability for a task to finish as constant in time.
For exponentials, and only exponentials, we do not need to keep track of how long a task has been running, which started first, etc. Each state is completely characterized by the set of tasks that are currently Active, Finished, or not yet started (Idle).

### Erlang Distributed Tasks

With slight admin, this basic framework can be augmented to allow each task to have multiple identical stages, that must finish in sequence. Under the hood, these are separate subtasks that must start upon the completion of the previous stage. However, it is more realistic as it allows
for tasks with an Erlang duration distribution, which notably doesn't have its mode at zero.

See an example of the distribution of three tasks, all with lambda=5, the first consists of 1 stage, the second of 2 stages, and the last of 5 stages. Changing lambda only scales the time axis.
![Erlang Graph](readme_imgs/erlang.png)

The price is of course added computational complexity.

## Execution Policies

Despite the resource requirements, and dependency requirements per task, there is still superexponential freedom in choosing the scheduling order. The most concise (but less powerful) way to encode our order preference, is with a _policy_.

A policy is simply a permutation of the task indices. I.e. for tasks (0,1,2), a policy could be [2,0,1]. Executing it means always starting the first idle task in the list that is allowed by the constraints. If no task is allowed, we wait until an active one finishes, then return to the policy. It is clear that there are n! policies for n tasks, though in execution they might not all be unique, due to constraints. 

They are also 'blind', not taking into account the history of the project. This is where the next section comes in.

### Contingent Stochastic Dynamic Programming

I have developed and coded a type of contingent Dijkstra's algorithm for the shortest traversal of the state-space graph from Project start to project finish. Let us call this technique Contingent Stochastic Dynamic Programming, or CSDP.
Executing a project in CSDP, we keep a lookup table of which tasks to start (or whether to wait), given that we find ourselves in any project state, such that the expected remaining duration is minimized. The reasoning hinges on linearity of expectations. Even if we are pushed off the expected course, because a task that should have taken long, takes only a short time, or vise versa, we can still use the lookup table to find the optimal next task to start, _contingent_ on inadvertently being in this state.

In the graph, this is represented by the red arrows. If a state has a red arrow leaving from it, then it is time-efficient to start that task as soon as we arrive at this state. We cannot predict which of the blue arrows will be traversed a priori (i.e. from FAA, we might end up in FFA or FAF), but we can always take a red one when presented.

How to use these tools: on to the next section.

## Getting Started

If you would simply like to play around with the repo out of the box, I suggest you install the `requirements.txt` file, and run `main.py`. It contains examples of the classes and how to interact with them, along with some comments.

## Modules in this Repo

The main modules are:

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

## Objects

### Task





## Sources
```
1: Cremers, S. (2015): Minimizing the expected makespan of a project with stochastic activity durations under resource constraints
2: Kulkarni, V., & Adlakha, V. (1986): Markov and Markov-regenerative PERT networks
```
