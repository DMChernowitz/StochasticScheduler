from typing import Dict, List, Union, Tuple

from src.Policy import Policy, DynamicPolicy
from src.Project import Project
from config import Config
from src.utils import get_title_string

import numpy as np
from statistics import NormalDist

import matplotlib.pyplot as plt


class Experiment:
    """An experiment consists of a project and a configuration, and can be run and analyzed.

    It allows us to run sampled policies on the same project many times and collect some statistics.
    """

    def __init__(self, experiment_config):
        """Create an experiment with a project and a configuration from a config object."""

        self.project: Project = Project.from_config(experiment_config)
        self.config: Config = experiment_config
        # dictionary with policy labels as keys and lists of completion times as values
        self.results_dict: Dict[str, List[int]] = {}

    def run(self):
        """Execute the policy many times, storing the statistics in self.results_dict.

        Policies are either labeled in terms of the precedence order of tasks, or as "CSDP".
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

        label = "CSDP"
        self.results_dict[label]: List[Union[int,float]] = []
        for n in range(self.config.n_runs):
            dynamic_policy = DynamicPolicy(self.project)
            timestep_finished = dynamic_policy.execute()
            self.results_dict[label].append(timestep_finished)

    def analyze(self):
        """Print the results of the experiment and show a histogram of the average completion times.

        Also report on the p-value of each policy being worse than CSDP.
        """
        print(self.project)
        if self.results_dict == {}:
            raise ValueError("Experiment has not been run yet")

        exp_str = get_title_string("Experiment Analysis")

        exp_str += f"After {self.config.n_runs} runs:\n"

        for label, p, better_worse in zip(self.results_dict.keys(), *self._calculate_p_value()):
            better = "better" if better_worse else "worse"
            significant = "significantly" if p < 0.05 else "insignificantly"
            exp_str += f"CSDP is {significant} {better} than policy {label} with p-value {round(p,5)}\n"

        exp_str += f"Comparison complete with {self.config.n_permutations} policies."
        print(exp_str+get_title_string(""))

        averages: List[float] = []
        min_dur = min([min(results) for results in self.results_dict.values()])
        max_dur = max([max(results) for results in self.results_dict.values()])
        n_bins = int(max(20,min(100, Config.n_runs/20)))
        bins = np.linspace(min_dur, max_dur, n_bins)

        label_used = False
        for label, results in self.results_dict.items():
            if label != "CSDP":
                averages.append(sum(results)/Config.n_runs)
                color = "blue"
                if label_used:
                    fig_label = None
                else:
                    fig_label = f"static policy (x {len(self.results_dict)-1})"
                    label_used = True
            else:
                color = "red"
                fig_label = "CSDP"
            plt.hist(results, label=fig_label, alpha=0.5, color=color, bins=bins)

        CSDP_average = sum(self.results_dict["CSDP"])/Config.n_runs
        shown_label = False
        for average in averages:
            if not shown_label:
                label = "Static policy average"
                shown_label = True
            else:
                label = None
            plt.axvline(x = average, color = "blue", label=label)
        plt.axvline(x = CSDP_average, color = "red", label="CSDP averages")
        plt.title(f"Distribution and avg of completion time (N={Config.n_runs}) of policies vs. CSDP")
        plt.xlabel("Completion time")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def _calculate_p_value(self) -> Tuple[List[float], List[bool]]:
        """Calculate the p-value of each individual policy being worse than CSDP.

        Makes the approximation that the t distribution can be approximated by a normal distribution.
        """
        if self.results_dict == {}:
            raise ValueError("Experiment has not been run yet")
        CSDP_average = sum(self.results_dict["CSDP"])/Config.n_runs
        static_averages = [sum(results)/Config.n_runs for label, results in self.results_dict.items() if label != "CSDP"]
        CSDP_variance = np.var(self.results_dict["CSDP"])
        static_variances = [np.var(results) for label, results in self.results_dict.items() if label != "CSDP"]

        # welch's t-test for each policy
        p_values = []
        CSDP_wins = []

        # For large N, the t distribution can be approximated by a normal distribution

        normaldist = NormalDist()

        for static_average, static_variance in zip(static_averages, static_variances):
            CSDP_wins.append(static_average > CSDP_average)
            n = Config.n_runs
            t = (CSDP_average - static_average) / np.sqrt(static_variance/n + CSDP_variance/n)
            if not CSDP_wins[-1]:
                t = -t
            p = normaldist.cdf(t)

            # to improve accuracy, install scipy and use the following code instead of the line above
            # df = (static_variance + CSDP_variance)**2 / (static_variance**2/(n-1) + CSDP_variance**2/(n-1))
            # p = scipy.stats.t.cdf(t, df)
            # one-sided tailed test: only care if static is worse than CSDP

            p_values.append(p)

        return p_values, CSDP_wins




