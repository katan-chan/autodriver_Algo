"""Utilities for building edge lists and index matrices."""

import numpy as np
from numba import njit


@njit
def build_edge_list_and_index(adjacency_bandwidth: np.ndarray) -> tuple:
    """Construct edge lists and index matrix from bandwidth adjacency."""

    n_nodes = adjacency_bandwidth.shape[0]

    m = 0
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if adjacency_bandwidth[u, v] > 0.0:
                m += 1

    edge_u = np.empty(m, dtype=np.int64)
    edge_v = np.empty(m, dtype=np.int64)
    edge_bandwidth = np.empty(m, dtype=np.float64)
    edge_index_matrix = np.full((n_nodes, n_nodes), -1, dtype=np.int64)

    idx = 0
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if adjacency_bandwidth[u, v] > 0.0:
                edge_u[idx] = u
                edge_v[idx] = v
                edge_bandwidth[idx] = adjacency_bandwidth[u, v]
                edge_index_matrix[u, v] = idx
                edge_index_matrix[v, u] = idx
                idx += 1

    return edge_u, edge_v, edge_bandwidth, edge_index_matrix
