from typing import List, Tuple, Union, Any, Dict

import numpy as np

import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

# the quantile of the exponential distribution that coincides with its mean
EXPONENTIAL_AVERAGE_QUANTILE = 1 - np.exp(-1)

# set to True to print more information from policy and project
VERBOSE = False

# the quantile to use when asking the max of continuous distributions with unbounded support.
MAX_DURATION_QUANTILE = 0.999


def get_title_string(title: str) -> str:
    """Return a centered title string with a line of dashes left and right."""
    total_len = 72
    if title:
        padded_title = (" " + title + ": ").center(total_len, "-")
    else:
        padded_title = "-" * total_len
    return f"\n{padded_title}\n"


def str_of_length(s: Any, length: int) -> str:
    """Return a string of length length, padding with spaces if necessary on the right.

    :param s: the string to pad
    :param length: the length of the output string
    """
    _s = str(s)[:length]
    return _s + " " * (length - len(_s))


def format_table(table: List[List[str]]) -> str:
    """Put together a table where the first row is the titles.

    :param table: a list of lists, where the first list is the titles of the columns
        all subsequent lists are the rows of the table
    """
    row_iter = range(len(table))
    col_iter = range(len(table[0]))
    col_widths = [max(len(table[_r][_c]) for _r in row_iter) + 1 for _c in col_iter]
    new_table = ["| ".join([str_of_length(table[_r][_c], col_widths[_c]) for _c in col_iter]) for _r in row_iter]
    hline = "|-".join(["-" * col_widths[_c] for _c in col_iter])
    new_table.insert(1, hline)
    new_table.append(hline)
    return "| " + "|\n| ".join(new_table) + "|"


def binom(a: int, b: int) -> int:
    """Binomial coefficient, a choose b.

    :param a: the total number of elements
    :param b: the number of elements to choose

    :return: the number of ways to choose b elements from a
    """
    r = 1
    for i in range(b):
        r *= a - i
        r //= i + 1
    return r


class HypoExponential:
    """Class to represent a sum of exponential distributions with distinct lambdas."""

    def __init__(self, lambdas: List[float]):
        """Initialize the distribution with a list of lambdas. Returns an object that can be called with
        a scalar or vector x to get the probability density at x."""
        if len(set(lambdas)) != len(lambdas):
            raise ValueError("Lambdas must be distinct")
        self.lambdas = np.array(lambdas, dtype=float)
        self.n = len(lambdas)
        self.mean = sum(1 / self.lambdas)
        self.variance = sum(self.lambdas ** -2)

        dif = self.lambdas[:, np.newaxis] - self.lambdas[np.newaxis, :] + np.eye(self.n)
        # precompute the weights for the distribution
        self.Ws = np.prod(self.lambdas[:, np.newaxis] / dif, axis=0)

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Return the probability density at x. Can be called with a scalar or a vector x."""
        if isinstance(x, np.ndarray):
            return np.sum(self.Ws[np.newaxis, :] * np.exp(-self.lambdas[np.newaxis, :] * x[:, np.newaxis]), axis=1)
        return sum(self.Ws * np.exp(-self.lambdas * x))


class Erlang:
    """Class to represent an Erlang distribution with parameters k and lambda:
    the sum of k exponential distributions."""

    def __init__(self, k: int, lam: float):
        """Initialize the distribution with parameters k and lambda. Returns an object that can be called with
        a scalar or vector x to get the probability density at x."""
        self.k = k
        self.lam = lam
        self.mean = k / lam
        self.variance = k / lam ** 2

        # precompute the constant for the distribution
        self.C = lam ** k / np.math.factorial(k - 1)

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Return the probability density at x. Can be called with a scalar or a vector x."""
        return self.C * x ** (self.k - 1) * np.exp(-self.lam * x)


# the following functions turns out not to be useful in practice, as it is biased towards very high k.
def moments_to_erlang(mu: float, var: float) -> Tuple[int, float]:
    """Get the integer k and lambda for an Erlang distribution with given mean and variance.

    mu = k/lambda
    var = k/lambda^2
    """
    # first get the closest integer k > 1, as physical tasks never have the mode at 0
    k = max(2, round(mu ** 2 / var))

    # then get the lambda that fits the mean best
    lam = k / mu
    return k, lam


class ArrowCoordMaker:
    """
    Helper class to take positions of states on the state space visualization and creates the coordinates of
    arrows between states if there is an allowed transition.

    Keeps track of how many vertical arrows have been made per row, and shifts them to not overlap"""

    def __init__(self, packing_factor: float = 0.8):
        """Initialize the ArrowCoordMaker with a packing factor, which determines how much space to leave between
        arrows that are in the same vertical position.

        Also initializes the progress_arrow_table, which keeps track of the arrows that have been made so far.
        New arrows are shifted to not overlap with existing arrows.
        """

        # key: x position, value: list of tuples (y_start, y_end, xorder)
        self.progress_arrow_table: Dict[int, List[Tuple[int, int, int]]] = {}
        self.bub = 0.1  # margin between end of the arrow and the state bubble (distance state-state is 1)
        self.packing_factor = packing_factor

    def make(
            self,
            start_pos: Tuple[int, int],
            end_pos: Tuple[int, int]
            ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
        """Create an arrow from start_pos to end_pos. Used in the visualization of the state space graph.

        :param start_pos: the x and y position of the start state
        :param end_pos: the x and y position of the end state

        :return: a tuple of two tuples: the first tuple is the coordinates of the arrow, the second is the midpoint
        """

        # first get all the arrows that are in the way, and find the lowest x shift (xorder) that is not taken
        if (vx := start_pos[0]) == end_pos[0]:
            self.progress_arrow_table.setdefault(vx, [])
            overlappers = []
            new_lower, new_upper = min(start_pos[1], end_pos[1]), max(start_pos[1], end_pos[1])
            for alt_lower, alt_upper, xorder in self.progress_arrow_table[vx]:
                if new_lower < alt_upper and new_upper > alt_lower:  # overlap.
                    # Two arrows can't completely coincide by construction.
                    overlappers.append(xorder)
            for xorder in range(len(self.progress_arrow_table[vx]) + 1):
                if xorder not in overlappers:
                    break
            self.progress_arrow_table[vx].append((new_lower, new_upper, xorder))
            shift_x = xorder * self.bub * self.packing_factor
        else:
            shift_x = 0

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dx_margin = self.bub if dx > 0.5 else 0
        if abs(dy) > 0.5:
            dy_margin = self.bub if dy > 0.5 else -self.bub
        else:
            dy_margin = 0
        arrow = (
            start_pos[0] + dx_margin + shift_x,  # x
            start_pos[1] + dy_margin,  # y
            dx - 2 * dx_margin,  # dx
            dy - 2 * dy_margin,  # dy
        )
        return arrow, (start_pos[0] + 0.5 * dx + shift_x, start_pos[1] + 0.5 * dy)


class HandlerEllipse(HandlerPatch):
    """Handler for the plot legend to show ellipses."""

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerArrow(HandlerPatch):
    """Handler for the plot legend to show arrows."""

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        p = mpatches.FancyArrow(0, 0.5 * height, width, 0, width=3, head_width=height,
                                head_length=0.5 * height, length_includes_head=True)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def plot_exponential_vs_erlang(
        lam: float = 5,
        ks: List[int] = None
) -> None:
    """Plot the exponential and several Erlang distributions for a fixed lambda and various k

    Meant to illustrate that the Erlang (sum of k exponentials)
    has a mode at a nonzero value of its independent parameter.

    :param lam: the lambda of the exponential distribution and the Erlang distributions
    :param ks: the k of the Erlang distribution
    """
    if ks is None:
        ks = [2, 5]

    import matplotlib.pyplot as plt

    x = np.linspace(0, 2, 1000)

    def exponential(x, lam: float) -> float:
        return lam * np.exp(-lam * x)

    plt.plot(x, exponential(x, lam), label=f"Exponential λ={lam}", linestyle="--")
    for k in ks:
        er = Erlang(k, lam)
        plt.plot(x, er(x), label=f"Erlang k={k} λ={lam}", lw=k)

    plt.xlabel("Time between task start and finish")
    plt.ylabel("Probability density")
    plt.legend()
    plt.show()


class DiGraph:
    """Class to represent a directed graph. Only used to count topological orderings."""

    # Constructor
    def __init__(self, edges, N):
        """Initialize a directed graph from a list of edges and the number of vertices N.

        :param edges: a list of tuples (src, dest) representing the edges of the graph.
            Should be in [0,N) for all src and dest.
        :param N: the number of vertices in the graph
        """
        # A List of Lists to represent an adjacency list
        self.adj_list = [[] for _ in range(N)]

        # stores in-degree of a vertex
        # initialize in-degree of each vertex with 0
        self.indegree = [0] * N

        # add edges to the undirected graph
        for (src, dest) in edges:
            # add an edge from source to destination
            self.adj_list[src].append(dest)

            # count in-degree
            self.indegree[dest] += 1

        # initialize the count of the orderings
        self._n_topological_orders: int = None

    @property
    def n_topological_orders(self) -> int:
        """Return the number of different ways to traverse the graph, following the directed edges."""
        if self._n_topological_orders is None:
            self._n_topological_orders = 0
            self._count_topological_orders(path=[], discovered=[False] * len(self.indegree))
        return self._n_topological_orders

    def _count_topological_orders(
            self,
            path: List[int],
            discovered: List[bool]
    ) -> None:
        """Get the number of different ways to traverse the graph, following the directed edges."""

        N = len(self.adj_list)
        for v in range(N):
            # proceed only if in-degree of current node is 0 and
            # current node is not processed yet
            if self.indegree[v] == 0 and not discovered[v]:

                # for every adjacent vertex u of v,
                # reduce in-degree of u by 1
                for u in self.adj_list[v]:
                    self.indegree[u] -= 1

                # include current node in the path
                # and mark it as discovered
                path.append(v)
                discovered[v] = True

                # recur
                self._count_topological_orders(path=path, discovered=discovered)

                # backtrack: reset in-degree
                # information for the current node
                for u in self.adj_list[v]:
                    self.indegree[u] += 1

                # backtrack: remove current node from the path and
                # mark it as undiscovered
                path.pop()
                discovered[v] = False

        # count the topological order if
        # all vertices are included in the path
        if len(path) == N:
            self._n_topological_orders += 1
