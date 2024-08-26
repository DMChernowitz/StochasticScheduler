from typing import List, Tuple, Dict, Set, Any

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


def log_normal_to_erlang(mu, sigma):
    k = int(np.round(1/(np.exp(sigma**2)-1)))
    lam = np.exp(-mu-sigma**2/2)/(np.exp(sigma**2)-1)
    return k, lam