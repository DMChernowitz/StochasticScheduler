# Stochastic Resource-Constrained Project Scheduling Tools

This is a project used to explore, solve, and visualize SRCPSPs, or Stochastic Resource-Constrained Project Scheduling Problems.

## Statespace Graph Framework and Memoryless tasks

I borrow heavily from the literature as produced by Cremers (1), who continued the work of Kulkarni and Adlakha (2). They formulate the state space of the project as a graph, with transitions between states when a particular task starts or finishes.

See the graph below for a toy project with 3 tasks (0,1 and 2). Task 1 has task 0 as a prerequisite. The graph is made using this repo.

![Toy Graph](readme_imgs/project_graph.png)

This is particularly powerful thanks to the memoryless property of exponentially distributed random variables.
For these, and only these, we do not need to keep track of how long a task has been running, which started first, etc. The probability density for a task to finish is constant in time. Each state is completely characterized by the set of tasks that are currently Active, Finished, or not yet started (Idle).
With slight admin, this basic framework can be augmented to allow each task to have multiple identical stages, that must finish in sequence. Under the hood, these are separate subtasks that must start upon the completion of the previous stage. However, it is more realistic as it allows
for tasks with an Erlang duration distribution, with notably doesn't have its mode at zero.

See an example of the distribution of three tasks, all with lambda=5, the first consists of 1 stage, the second of 2 stages, and the last of 5 stages. Changing lambda only scales the time axis.
![Erlang Graph](readme_imgs/erlang.png)

## Contingent Stochastic Dynamic Programming

In this formulation, we can run simulations of different Policies (more below). I have also developed and coded a type of contingent Dijkstra's algorithm for the shortest traversal of the state-space graph from Project start to project finish. Let us call this technique Contingent Stochastic Dynamic Programming, or CSDP.
Executing a project in CSDP, we keep a lookup table of which tasks to start (or whether to wait), given that we find ourselves in any project state, such that the expected remaining duration is minimized. The reasoning hinges on linearity of expectations. Even if we are pushed off the expected course, because a task that should have taken long, takes only a short time, or vise versa, we can still use the lookup table to find the optimal next task to start, _contingent_ on inadvertently being in this state.

In the graph, this is represented by the red arrows. If a state has a red arrow leaving from it, then it is time-efficient to start that task as soon as we arrive at this state. We cannot predict which of the blue arrows will be traversed a priori (i.e. from FAA, we might end up in FFA or FAF), but we can always take a red one when presented.

## Modules in this Repo

- Project.py
- Policy.py
- Experiment.py
- StateSpace.py

Which all borrow from the Objects.py classes:

- Task
- Resource
- ProbabilityDistribution

```
Sources:
1: Cremers, S. (2015): Minimizing the expected makespan of a project with stochastic activity durations under resource constraints
2: Kulkarni, V., & Adlakha, V. (1986): Markov and Markov-regenerative PERT networks
```
