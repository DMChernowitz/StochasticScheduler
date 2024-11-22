class Config:
    # project parameters
    max_simultaneous_resources_required = 8
    resource_available = 10
    duration_average_range = (2,7)
    duration_variance_range = (5,10)
    n_tasks = 3
    max_dependencies = 1
    prob_type = "erlang"
    max_duration_quantile= 0.999
    max_stages = 1

    # experiment parameters
    n_runs = 800 # for simulation per policy
    n_permutations = 5 # for policies (lists of priorities)