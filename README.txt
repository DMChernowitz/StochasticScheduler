Exponential Dijkstra Scheduler

A small POC project to explore Stochastic Resource Constrained Project Scheduling.

It has four main components
- A Project that consists of a set of tasks, with probabilistic durations, and renewable resources available.
    - Tasks can be dependent on other tasks
    - Tasks can be parallelized
    - Tasks have resource requirements
    - Tasks can have exponential distributions or discrete distributions
    - A project can be created from a config file
- A Policy, which is a prioritization strategy for tasks. The Policy takes a project and can simulate execution.
    - Tasks are executed in the order they appear in the policy, as long as their dependencies are met and resources are available.
    - By simulating execution many times, we can get a distribution of project durations,
- A StateSpace class, a data structure that holds possible states of the project, and has a stochastic dynamic programming algorithm to find the optimal policy.
    - The state space is a directed graph, where each node is a state of the project, and each edge is a possible transition between states.
    - Moving through the state space can happen deterministically or stochastically. The former when starting a task. The latter whenever we finish a task.
- A DynamicPolicy class which uses the SDP to find an optimal, dynamic policy.
    - At any point in the project, we start tasks or wait. These decisions follow the shortest stochastic path to completion.

Also included are
- An Experiment class that runs a simulation of a project with a given policy, and returns the duration of the project.
- Two probability distributions, including exponential, discrete integer. Full functionality is not implemented for discrete distributions, as the state space is too large.
- A generator function to randomly product tasks with dependencies and resources.
- A snipping function that reduces the dependencies of each task to the minimal set (removing dependencies that are implied by other dependencies).

An interesting observation is that with exponentially distributed tasks, the optimal policy is the same if we consider expected durations or any quantile of the distribution (such as the median). This is not true in general.