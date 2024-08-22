from src.Experiment import Experiment
from src.config import Config
from src.Policy import Policy

import numpy as np


if __name__ == '__main__':

    config: Config = Config()

    # create an experiment, consisting of a specific project and a specific configuration
    experiment: Experiment = Experiment(config)
    # run the experiment
    experiment.run()
    # analyze the results
    experiment.analyze()

    # create a random policy
    base_policy = list(range(config.n_tasks))
    np.random.shuffle(base_policy)

    # execute the policy and show the results as an example
    policy = Policy(experiment.project, base_policy)
    policy.execute()
    print(policy)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
