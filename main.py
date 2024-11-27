from src.Experiment import Experiment
from src.config import RandomConfig, LiteralConfig
from src.Policy import Policy
from src.Project import Project

if __name__ == '__main__':

    config: LiteralConfig = LiteralConfig()

    print("Let's create a canned project and check out its state space!\n")

    # execute the policy and show the results as an example
    project = Project.from_config(config)

    # show a graph of the allowed transitions of the project
    project.visualize_state_space(metastate_mode=True, rich_annotations=True)

    print("Interestingly, we could start task 1 early on, but it's faster to wait for task 0 and 2 to finish first.")

    # print the contingency table
    print("\n\n\n\n\nWe can also print the contingency table of the project for CSDP!\n")
    project.print_contingency_table()

    print("\n\n\n\n\nLet's carry out our project with a random policy as a demo")

    # create an arbitrary policy: a preferred order of tasks
    base_policy = [2, 0, 1, 3]

    # initialize the policy
    policy = Policy(project, base_policy)
    # and execute: determine when tasks start and end (in one realization)
    policy.execute()

    print("\n\n\n\n\nLet's see the details of the policy and how it functioned on this project!\n")
    print(policy)

    print("\n\n\n\n\n Let's run an experiment on a large sample of policies!\n")
    print("The experiment is meant to compare classical, static policies with a dynamic policy.\n")

    # create an experiment, consisting of a specific project and a specific configuration
    experiment: Experiment = Experiment(config)
    # run the experiment, i.e. carry out the policies many times and collect statistics
    experiment.run()
    # analyze the results, also creates a histogram of the completion times.
    experiment.analyze()


