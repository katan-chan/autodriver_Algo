"""Utilities for building edge lists and index matrices."""

import numpy as np


def build_edge_list_and_index(adjacency_bandwidth: np.ndarray) -> tuple:
    """Construct edge lists and index matrix from bandwidth adjacency."""

    n_nodes = adjacency_bandwidth.shape[0]
    edges = np.where(np.triu(adjacency_bandwidth, k=1) > 0.0)
    u_list, v_list = edges
    m = u_list.shape[0]

    edge_u = np.empty(m, dtype=np.int64)
    edge_v = np.empty(m, dtype=np.int64)
    edge_bandwidth = np.empty(m, dtype=np.float64)
    edge_index_matrix = np.full((n_nodes, n_nodes), -1, dtype=np.int64)

    for idx, (u, v) in enumerate(zip(u_list, v_list)):
        bandwidth = adjacency_bandwidth[u, v]
        edge_u[idx] = u
        edge_v[idx] = v
        edge_bandwidth[idx] = bandwidth
        edge_index_matrix[u, v] = idx
        edge_index_matrix[v, u] = idx

    return edge_u, edge_v, edge_bandwidth, edge_index_matrix
