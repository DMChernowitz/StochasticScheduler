from typing import Dict, List, Union

from src.Policy import Policy, DynamicPolicy
from src.Project import Project
from src.config import Config

import numpy as np
from statistics import NormalDist

import matplotlib.pyplot as plt


class Experiment:
    """An experiment consists of a project and a configuration, and can be run and analyzed.

    It allows us to run sampled policies on the same project many times and collect some statistics.
    """

    def __init__(self, config):

        self.project: Project = Project.from_config(config)
        self.config: Config = config
        # dictionary with policy labels as keys and lists of completion times as values
        self.results_dict: Dict[str, List[int]] = {}

    def run(self):
        """Execute the policy many times, storing the statistics in self.results_dict.

        Policies are either labeled in terms of the precedence order of tasks, or as "dijkstra".
        The execution is probabilistic, so the results will vary between runs. We run each policy
        self.config.n_runs times.
        """
        base_policy_list = list(range(len(self.project.task_list)))

        for _ in range(self.config.n_permutations):

            policy_list = base_policy_list.copy()
            np.random.shuffle(policy_list)

            label: str = str(policy_list)
            self.results_dict[label]: List[Union[int,float]] = []

            for n in range(self.config.n_runs):
                policy = Policy(self.project.reset_task_stages(), list(policy_list))

                timestep_finished = policy.execute()
                # print(f"Run {n} finished after {timestep_finished} timesteps with policy {label}")
                # print(policy.state_sequence)

                self.results_dict[label].append(timestep_finished)

        label = "dijkstra"
        self.results_dict[label]: List[Union[int,float]] = []
        for n in range(self.config.n_runs):
            dynamic_policy = DynamicPolicy(self.project)
            timestep_finished = dynamic_policy.execute()
            self.results_dict[label].append(timestep_finished)

    def analyze(self):
        """Print the results of the experiment and show a histogram of the average completion times.

        Also report on the p-value of each policy being worse than dijkstra.
        """
        print(self.project)
        if self.results_dict == {}:
            raise ValueError("Experiment has not been run yet")
        print(f"Analyzing experiment after {self.config.n_runs} runs:")

        for label, p, better_worse in zip(self.results_dict.keys(), *self.calculate_p_value()):
            better = "better" if better_worse else "worse"
            significant = "significantly" if p < 0.05 else "insignificantly"
            print(f"Dijkstra is {significant} {better} than policy {label} with p-value {round(p,5)}")

        averages: List[float] = []
        min_dur = min([min(results) for results in self.results_dict.values()])
        max_dur = max([max(results) for results in self.results_dict.values()])
        n_bins = int(max(20,min(100, Config.n_runs/20)))
        bins = np.linspace(min_dur, max_dur, n_bins)

        label_used = False
        for label, results in self.results_dict.items():
            if label != "dijkstra":
                averages.append(sum(results)/Config.n_runs)
                color = "blue"
                if label_used:
                    fig_label = None
                else:
                    fig_label = f"static policy (x {len(self.results_dict)-1})"
                    label_used = True
            else:
                color = "red"
                fig_label = "dijkstra"
            plt.hist(results, label=fig_label, alpha=0.5, color=color, bins=bins)

        dijkstra_average = sum(self.results_dict["dijkstra"])/Config.n_runs
        shown_label = False
        for average in averages:
            if not shown_label:
                label = "static policy average"
                shown_label = True
            else:
                label = None
            plt.axvline(x = average, color = "blue", label=label)
        plt.axvline(x = dijkstra_average, color = "red", label="dijkstra averages")
        plt.title(f"Distribution and avg of completion time (N={Config.n_runs}) of policies vs. Dijkstra")
        plt.xlabel("Completion time")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def calculate_p_value(self):
        """Calculate the p-value of each individual policy being worse than dijkstra.

        Makes the approximation that the t distribution can be approximated by a normal distribution.
        """
        if self.results_dict == {}:
            raise ValueError("Experiment has not been run yet")
        dijkstra_average = sum(self.results_dict["dijkstra"])/Config.n_runs
        static_averages = [sum(results)/Config.n_runs for label, results in self.results_dict.items() if label != "dijkstra"]
        dijkstra_variance = np.var(self.results_dict["dijkstra"])
        static_variances = [np.var(results) for label, results in self.results_dict.items() if label != "dijkstra"]

        # welch's t-test for each policy
        p_values = []
        dijkstra_wins = []

        # For large N, the t distribution can be approximated by a normal distribution

        normaldist = NormalDist()

        for static_average, static_variance in zip(static_averages, static_variances):
            dijkstra_wins.append(static_average > dijkstra_average)
            n = Config.n_runs
            t = (dijkstra_average - static_average) / np.sqrt(static_variance/n + dijkstra_variance/n)
            if not dijkstra_wins[-1]:
                t = -t
            p = normaldist.cdf(t)

            # to improve accuracy, install scipy and use the following code instead of the line above
            # df = (static_variance + dijkstra_variance)**2 / (static_variance**2/(n-1) + dijkstra_variance**2/(n-1))
            # p = scipy.stats.t.cdf(t, df)
            # one-sided tailed test: only care if static is worse than dijkstra

            p_values.append(p)

        return p_values, dijkstra_wins




