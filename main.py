from src.Experiment import Experiment
from src.config import RandomConfig, LiteralConfig
from src.Policy import Policy
from src.Project import Project

import numpy as np

# Call these child-tasks, that always follow their parent tasks.
# A task then points to a child task.. pointing forward instead of backward with dependencies.
# child tasks must have the same lambdas, dependencies...
# they should be in the same Task instance, but with a stage counter.
# that means their states, when active, are not just status 1, but 1a, 1b, 1c, etc? Think of an elegant way to do this.
# I guess replacing the 1 with a tuple (1,0) would work, or with a letter...
# In the policy framework, child tasks don't get a number / order,
# but are always started after their parent task finishes.
# This allows for modelling of atomic tasks that can't be broken into subtasks, but with still a with a minimum duration
# using exponential distributions.
# We model tasks with Erlang distributions. Use the moments_to_erlang function to convert moments to Erlang parameters.

# Erlang has the smallest variance of all hypoexponentials with the same mean.

# if __name__ == '__main__':
#
#     from src.utils import Erlang
#     import matplotlib.pyplot as plt
#
#     lambdas = [5]
#
#     x = np.linspace(0, 2, 1000)
#
#     def exponential(x, lam: float) -> float:
#         return lam * np.exp(-lam * x)
#
#
#     for l in lambdas:
#         plt.plot(x, exponential(x, l), label=f"Exponential λ={l}", linestyle="--")
#         for k in [2, 5]:
#             er = Erlang(k, l)
#             plt.plot(x, er(x), label=f"Erlang k={k} λ={l}", lw=k)
#
#     plt.xlabel("Time between task start and finish")
#     plt.ylabel("Probability density")
#     plt.legend()
#     plt.show()

if __name__ == '__main__':

    config: RandomConfig = RandomConfig()

    print("Let's create a random project and check out its state space!\n")

    # execute the policy and show the results as an example
    project = Project.from_config(config)

    # config2: LiteralConfig = LiteralConfig()
    # project2 = Project.from_config(config2)

    # show a graph of the allowed transitions of the project
    project.visualize_state_space(metastate_mode=False, rich_annotations=True)

    # print the contingency table
    print("\n\n\n\n\nWe can also print the contingency table of the project for CSDP!\n")
    project.print_contingency_table()

    print("\n\n\n\n\nLet's carry out our project with a random policy as a demo")

    # create a random policy
    base_policy = list(range(config.n_tasks))
    np.random.shuffle(base_policy)

    # initialize the policy
    policy = Policy(project, base_policy)
    # and execute: determine when tasks start and end (in one realization)
    policy.execute()

    print("\n\n\n\n\nLet's see the details of the policy and how it functioned on this project!\n")
    print(policy)

    print("\n\n\n\n\n Let's run an experiment on a large sample of policies!\n")

    # create an experiment, consisting of a specific project and a specific configuration
    experiment: Experiment = Experiment(config)
    # run the experiment, i.e. carry out the policies many times and collect statistics
    experiment.run()
    # analyze the results, also creates a histogram of the completion times.
    experiment.analyze()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
