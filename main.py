from src.Experiment import Experiment
from src.config import Config
from src.Policy import Policy
from src.Project import Project

import numpy as np


if __name__ == '__main__':

    config: Config = Config()

    print("Let's create a project and check out its state space!\n")

    # execute the policy and show the results as an example
    project = Project.from_config(config)

    # show a graph of the allowed transitions of the project
    project.visualize_state_space()

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
