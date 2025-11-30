"""Dijkstra shortest path implementation using dense adjacency matrix."""

import numpy as np
from numba import njit


@njit
def dijkstra_shortest_path(
    adjacency_travel_time: np.ndarray,
    source: int,
    target: int,
) -> tuple:
    """Compute single-source shortest path using O(n^2) Dijkstra."""

    n_nodes = adjacency_travel_time.shape[0]
    dist = np.full(n_nodes, np.inf)
    prev = np.full(n_nodes, -1, dtype=np.int64)
    visited = np.zeros(n_nodes, dtype=np.bool_)

    dist[source] = 0.0

    for _ in range(n_nodes):
        u = -1
        min_val = np.inf
        for i in range(n_nodes):
            if (not visited[i]) and (dist[i] < min_val):
                min_val = dist[i]
                u = i

        if u == -1 or u == target:
            break

        visited[u] = True
        row = adjacency_travel_time[u]
        for v in range(n_nodes):
            weight = row[v]
            if weight == np.inf:
                continue
            alt = dist[u] + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u

    path = np.full(n_nodes, -1, dtype=np.int64)
    if dist[target] == np.inf:
        return dist[target], path

    tmp = np.full(n_nodes, -1, dtype=np.int64)
    idx = 0
    cur = target
    while cur != -1 and idx < n_nodes:
        tmp[idx] = cur
        cur = prev[cur]
        idx += 1

    for i in range(idx):
        path[i] = tmp[idx - 1 - i]

    return dist[target], path
