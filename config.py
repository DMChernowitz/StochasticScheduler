class Config:
    """Parent class for both types of project configs.

    Also used to store the parameters of the simulation and the experiment."""
    # experiment parameters
    n_runs = 800 # for simulation per policy
    n_permutations = 5  # for policies (lists of priorities)

    # for significance testing
    p_value_threshold = 0.05

    resource_capacities: dict = {
        "centrifuge": 9,
        "crane": 16,
        "drill": 10,
    }  # number of resources available for each type. Can also be an int, then all resources have the same capacity.


class RandomConfig(Config):
    """Class used to generate projects with random, but somewhat realistic, settings."""

    n_tasks: int = 10
    max_dependencies: int = 2  # per task. Dependencies are chosen randomly from lower task_ids, up to this number.

    max_simultaneous_resources_required: int = 8  # per task per resource type

    minimum_duration_range = (1,4)  # Interval from which to choose the minimum duration of a task uniformly.
    # not used if prob_type is "exponential"

    duration_spread_range = (1,5)  # Interval from which to choose difference between min and max duration uniformly
    # if prob_type is "exponential", uses duration_spread_range as the interval from which to uniformly sample the mean

    # can be "exponential", "binomial", "uniform", or "random
    prob_type = "exponential"
    max_stages = 3  # choose uniformly from 1 to max_stages, per task


class LiteralConfig(Config):
    """Class used to generate projects with specific settings, chosen by hand."""
    tasks = [
        dict(
            id=0,
            dependencies=[],
            distribution="exponential",
            stages=2,
            avg_stage_duration=4,
            resource_requirements={"drill": 5, "centrifuge": 5}
        ),
        dict(
            id=1,
            dependencies=[],
            distribution="exponential",
            stages=3,
            avg_stage_duration=7,
            resource_requirements={"drill": 2, "crane": 9}
        ),
        dict(
            id=2,
            dependencies=[0],
            distribution="exponential",
            stages=2,
            avg_stage_duration=3,
            resource_requirements={"crane": 10}
        ),
        dict(
            id=3,
            dependencies=[2],
            distribution="exponential",
            stages=1,
            avg_stage_duration=11,
            resource_requirements={"centrifuge": 6}
        ),
    ]