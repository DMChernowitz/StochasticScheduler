class Config:
    # technical parameters
    max_duration_quantile = 0.999

    # experiment parameters
    n_runs = 800 # for simulation per policy
    n_permutations = 5 # for policies (lists of priorities)


class RandomConfig(Config):
    # random project parameters
    max_simultaneous_resources_required = 8  # per task per resoruce type
    resource_available = 10  # for each resource type
    duration_average_range = (2,7)  # just a heuristic indication of the final result. Will vary by prob_type
    duration_variance_range = (5,10)  # just a heuristic indication of the final result. Will vary by prob_type
    n_tasks = 3
    max_dependencies = 1 # per task
    # can be "erlang", "exponential", "binomial", "uniform", or "random
    prob_type = "erlang"
    max_stages = 2  # per task


class LiteralConfig(Config):
    resource_capacities = dict(
        centrifuge=10,
        crane=10,
        drill=10,
    )
    tasks = [
        dict(
            id=0,
            dependencies=[],
            distribution="erlang",
            stages=2,
            avg_stage_duration=4,
            resource_requirements={"drill": 5, "centrifuge": 5}
        ),
        dict(
            id=1,
            dependencies=[],
            distribution="erlang",
            stages=3,
            avg_stage_duration=7,
            resource_requirements={"drill": 2, "crane": 9}
        ),
        dict(
            id=2,
            dependencies=[0],
            distribution="erlang",
            stages=2,
            avg_stage_duration=3,
            resource_requirements={"crane": 10}
        ),
        dict(
            id=3,
            dependencies=[2],
            distribution="erlang",
            stages=1,
            avg_stage_duration=11,
            resource_requirements={"centrifuge": 6}
        ),
    ]


class DijkstraConfig(Config):
    resource_capacities = dict(
        centrifuge=10,
        crane=10,
        drill=10,
    )
    tasks = [
        dict(
            id=0,
            dependencies=[],
            distribution="erlang",
            stages=2,
            avg_stage_duration=2,
            resource_requirements={"drill": 5, "centrifuge": 5}
        ),
        dict(
            id=1,
            dependencies=[],
            distribution="erlang",
            stages=1,
            avg_stage_duration=3,
            resource_requirements={"drill": 2, "crane": 9}
        ),
    ]