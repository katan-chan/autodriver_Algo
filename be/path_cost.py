"""Utilities for computing travel cost of a path."""

import numpy as np
from numba import njit


@njit
def compute_path_travel_cost(path: np.ndarray, adjacency_travel_time: np.ndarray) -> float:
    """Sum travel time along a padded path sequence."""

    n_nodes = path.shape[0]
    total = 0.0
    for i in range(n_nodes - 1):
        u = path[i]
        v = path[i + 1]
        if v < 0:
            break
        weight = adjacency_travel_time[u, v]
        if weight == np.inf:
            return np.inf
        total += weight

    return total
