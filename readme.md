This is a project used to explore SRCPSPs, or Stochastic Resource-Constrained Project Scheduling Problems.

We borrow heavily from the literature as produced by Cremers, who continued the work of Kulkarni and Adlakha. They formulates the state space of the project as a graph. This is particularly powerful thanks to the memoryless property of exponentially distributed random variables.
For these, and only these, we do not need to keep track of how long a task has been running. Each state is completely characterized by the set of tasks that are currently running, completed, or not yet started.
On this basic framework, we can augment with slightly more administration, to allow each task to have multiple stages. Under the hood, these are separate subtasks that must start upon the completion of the previous stage. However, it is more realistic as it allows
for tasks with an Erlang duration distribution, with notably doesn't have its mode at zero.

In this formulation, we can run simulations of different Policies (more below), but we have also developed and coded a type of contigent Dijkstra's algorithm for the shortest traversal of the state-space graph from Project start to project finish. Let us call this technique Contingent Stochastic Dynamic Programming, or CSDP.
Executing a project in CSDP, we keep a lookup table of which tasks to start, given that we find ourselves in any project state, such that the expected remaining duration is minimized. It uses linearity of expectations. Even if we are pushed off the expected course, because a task that should have taken long, takes only a short time, or vise versa, we can still use the lookup table to find the _contingent_ optimal next task to start.

Sources:
- Cremers, S. (2015): Minimizing the expected makespan of a project with stochastic activity durations under resource constraints
- Kulkarni, V., & Adlakha, V. (1986): Markov and Markov-regenerative PERT networks