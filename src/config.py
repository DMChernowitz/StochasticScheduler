class Config:
    # project parameters
    max_simultaneous_resources_required = 8
    resource_available = 10
    min_days_range = 4
    max_days_range = 4
    n_tasks = 15
    prob_type = "exponential"
    max_duration_quantile= 0.999

    # experiment parameters
    n_runs = 800 # for simulation per policy
    n_permutations = 5 # for policies (lists of priorities)