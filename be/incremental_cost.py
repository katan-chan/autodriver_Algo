"""Penalty-aware cost computation when assigning a path."""

import numpy as np
from numba import njit

from .time_slots import count_vehicles_in_interval


@njit
def compute_incremental_cost_for_path(
    path: np.ndarray,
    base_cost: float,
    edge_index_matrix: np.ndarray,
    edge_bandwidth: np.ndarray,
    edge_loads: np.ndarray,
    beta_penalty: float,
) -> float:
    """Compute travel + congestion penalty if one more vehicle uses the path."""

    if base_cost == np.inf:
        return np.inf

    total_penalty = 0.0
    for idx in range(path.shape[0] - 1):
        u = path[idx]
        w = path[idx + 1]
        if w < 0:
            break
        edge_idx = edge_index_matrix[u, w]
        if edge_idx < 0:
            continue
        bandwidth = edge_bandwidth[edge_idx]
        new_load = float(edge_loads[edge_idx]) + 1.0
        overflow_ratio = new_load / bandwidth - 1.0
        if overflow_ratio > 0.0:
            total_penalty += np.exp(beta_penalty * overflow_ratio)

    return base_cost + total_penalty


@njit
def compute_incremental_cost_time_aware(
    path: np.ndarray,
    base_cost: float,
    start_time: float,
    adjacency_travel_time: np.ndarray,
    edge_index_matrix: np.ndarray,
    edge_bandwidth: np.ndarray,
    edge_time_slots: np.ndarray,
    beta_penalty: float,
) -> float:
    """Compute penalty by counting concurrent loads only during traversal windows."""

    if base_cost == np.inf:
        return np.inf

    n_nodes = path.shape[0]
    total_penalty = 0.0
    current_time = start_time

    for idx in range(n_nodes - 1):
        u = path[idx]
        v = path[idx + 1]
        if v < 0:
            break

        edge_idx = edge_index_matrix[u, v]
        if edge_idx < 0:
            continue

        travel_time = adjacency_travel_time[u, v]
        if travel_time == np.inf:
            return np.inf

        entry_time = current_time
        exit_time = current_time + travel_time

        current_load = count_vehicles_in_interval(edge_time_slots, edge_idx, entry_time, exit_time)
        bandwidth = edge_bandwidth[edge_idx]
        new_load = float(current_load) + 1.0
        overflow_ratio = new_load / bandwidth - 1.0
        if overflow_ratio > 0.0:
            total_penalty += np.exp(beta_penalty * overflow_ratio)

        current_time = exit_time

    return base_cost + total_penalty
