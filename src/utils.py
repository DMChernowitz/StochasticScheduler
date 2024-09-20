from typing import List, Tuple, Union, Any

import numpy as np

def str_of_length(s: Any, length: int) -> str:
    """Return a string of length length, padding with spaces if necessary"""
    _s = str(s)[:length]
    return _s + " " * (length - len(_s))


def format_table(table: List[List[str]]) -> str:
    """Put together a table where the first row is the titles."""
    row_iter = range(len(table))
    col_iter = range(len(table[0]))
    col_widths = [max(len(table[_r][_c]) for _r in row_iter)+1 for _c in col_iter]
    new_table = ["| ".join([str_of_length(table[_r][_c],col_widths[_c]) for _c in col_iter]) for _r in row_iter]
    hline = "|-".join(["-"*col_widths[_c] for _c in col_iter])
    new_table.insert(1,hline)
    new_table.append(hline)
    return "| "+"|\n| ".join(new_table)+"|"


def binom(a,b):
    r = 1
    for i in range(b):
        r *= a-i
        r //= i+1
    return r


class HypoExponential:
    def __init__(self, lambdas: List[float]):
        if len(set(lambdas)) != len(lambdas):
            raise ValueError("Lambdas must be distinct")
        self.lambdas = np.array(lambdas, dtype=float)
        self.n = len(lambdas)
        self.mean = sum(1/self.lambdas)
        self.variance = sum(self.lambdas**-2)

        dif = self.lambdas[:,np.newaxis] - self.lambdas[np.newaxis,:] + np.eye(self.n)
        # precompute the weights for the distribution
        self.Ws = np.prod(self.lambdas[:,np.newaxis]/dif, axis=0)

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(x, np.ndarray):
            return np.sum(self.Ws[np.newaxis,:] * np.exp(-self.lambdas[np.newaxis,:] * x[:,np.newaxis]), axis=1)
        return sum(self.Ws * np.exp(-self.lambdas * x))


class Erlang:
    def __init__(self, k: int, lam: float):
        self.k = k
        self.lam = lam
        self.mean = k/lam
        self.variance = k/lam**2

        # precompute the constant for the distribution
        self.C = lam**k/np.math.factorial(k-1)

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.C * x**(self.k-1) * np.exp(-self.lam * x)


def hypo_exponential(x: float, lambdas: List[float]) -> float:
    y = 0
    n = len(lambdas)
    for i in range(n):
        W = 1
        for j in range(n):
            if i == j:
                continue
            W *= lambdas[j] / (lambdas[j] - lambdas[i])
        y += W * lambdas[i] * np.exp(-lambdas[i] * x)
    return y


def erlang_distribution(x: float, k: int, lam: float) -> float:
    return (lam**k * x**(k-1) * np.exp(-lam * x)) / np.math.factorial(k-1)


def log_normal(x: float, mu: float, sigma: float) -> float:
    return np.exp(-0.5 * ((np.log(x) - mu) / sigma)**2) / (x * sigma * np.sqrt(2 * np.pi))


def log_normal_mean(mu: float, sigma: float) -> float:
    return np.exp(mu + sigma**2 / 2)


def log_normal_variance(mu: float, sigma: float) -> float:
    return (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)


def erlang_closest_to_log_normal(mu: float, sigma: float) -> Tuple[int, float]:
    ln_mu = log_normal_mean(mu, sigma)
    ln_var = log_normal_variance(mu, sigma)
    k = int(np.round(ln_mu**2 / ln_var))
    lam = ln_mu/ln_var
    return k, lam


def moments_to_erlang(mu: float, var: float) -> Tuple[int, float]:
    """Get the integer k and lambda for an Erlang distribution with given mean and variance.

    mu = k/lambda
    var = k/lambda^2
    """
    # first get the closest integer k > 1, as physical tasks never have the mode at 0
    k = max(2, round(mu**2 / var))

    # then get the lambda that fits the mean best
    lam = k / mu
    return k, lam

