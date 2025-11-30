"""Penalty-aware cost computation when assigning a path."""

import numpy as np


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
