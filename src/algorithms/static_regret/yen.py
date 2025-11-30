"""Yen's algorithm implementation for K-shortest simple paths."""

import numpy as np
from numba import njit

from ..common.dijkstra import dijkstra_shortest_path
from ..common.path_cost import compute_path_travel_cost


@njit
def _is_same_path(path1: np.ndarray, path2: np.ndarray) -> bool:
    """Return True when both path arrays describe the same route."""

    for idx in range(path1.shape[0]):
        a = path1[idx]
        b = path2[idx]
        if a == -1 and b == -1:
            return True
        if a != b:
            return False
    return True


@njit
def _add_candidate(
    candidate: np.ndarray,
    cost: float,
    paths: np.ndarray,
    num_found: int,
    cand_paths: np.ndarray,
    cand_costs: np.ndarray,
    cand_used: np.ndarray,
) -> None:
    """Insert a new candidate path if it is unique and has finite cost."""

    if cost == np.inf:
        return

    for idx in range(num_found):
        if _is_same_path(candidate, paths[idx]):
            return

    max_candidates = cand_paths.shape[0]
    for idx in range(max_candidates):
        if cand_used[idx] and _is_same_path(candidate, cand_paths[idx]):
            return

    free_idx = -1
    for idx in range(max_candidates):
        if not cand_used[idx]:
            free_idx = idx
            break

    if free_idx == -1:
        return

    for j in range(candidate.shape[0]):
        cand_paths[free_idx, j] = candidate[j]
    cand_costs[free_idx] = cost
    cand_used[free_idx] = True


@njit
def _pop_best_candidate(
    cand_paths: np.ndarray, cand_costs: np.ndarray, cand_used: np.ndarray
) -> tuple[int, float]:
    """Return index and cost of the best current candidate, or (-1, inf)."""

    max_candidates = cand_paths.shape[0]
    best_idx = -1
    best_cost = np.inf
    for idx in range(max_candidates):
        if cand_used[idx] and cand_costs[idx] < best_cost:
            best_cost = cand_costs[idx]
            best_idx = idx
    return best_idx, best_cost


@njit
def yen_k_shortest_paths(
    adjacency_travel_time: np.ndarray,
    source: int,
    target: int,
    k_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return K shortest simple paths (costs, paths) using Yen's algorithm."""

    n_nodes = adjacency_travel_time.shape[0]
    paths = np.full((k_paths, n_nodes), -1, dtype=np.int64)
    costs = np.full(k_paths, np.inf)

    max_candidates = k_paths * n_nodes
    cand_paths = np.full((max_candidates, n_nodes), -1, dtype=np.int64)
    cand_costs = np.full(max_candidates, np.inf)
    cand_used = np.zeros(max_candidates, dtype=np.bool_)

    base_cost, base_path = dijkstra_shortest_path(adjacency_travel_time, source, target)
    if base_cost == np.inf:
        return costs, paths

    for j in range(n_nodes):
        paths[0, j] = base_path[j]
    costs[0] = base_cost
    num_found = 1

    for k in range(1, k_paths):
        if costs[k - 1] == np.inf:
            break
        prev_path = paths[k - 1]
        path_len = 0
        for idx in range(n_nodes):
            if prev_path[idx] == -1:
                break
            path_len += 1
        if path_len == 0:
            break

        for spur_idx in range(path_len - 1):
            spur_node = prev_path[spur_idx]
            adj_tmp = adjacency_travel_time.copy()

            for a_idx in range(num_found):
                path_a = paths[a_idx]
                same_prefix = True
                for idx in range(spur_idx + 1):
                    if path_a[idx] != prev_path[idx]:
                        same_prefix = False
                        break
                if same_prefix:
                    next_node = path_a[spur_idx + 1]
                    if next_node != -1:
                        adj_tmp[spur_node, next_node] = np.inf
                        adj_tmp[next_node, spur_node] = np.inf

            for prefix_idx in range(spur_idx):
                ban_node = prev_path[prefix_idx]
                if ban_node == -1:
                    break
                for v in range(n_nodes):
                    adj_tmp[ban_node, v] = np.inf
                    adj_tmp[v, ban_node] = np.inf

            spur_cost, spur_path = dijkstra_shortest_path(adj_tmp, spur_node, target)
            if spur_cost == np.inf:
                continue

            new_path = np.full(n_nodes, -1, dtype=np.int64)
            pos = 0
            for idx in range(spur_idx + 1):
                new_path[pos] = prev_path[idx]
                pos += 1
            for idx in range(1, n_nodes):
                node_sp = spur_path[idx]
                if node_sp == -1 or pos >= n_nodes:
                    break
                new_path[pos] = node_sp
                pos += 1

            total_cost = compute_path_travel_cost(new_path, adjacency_travel_time)
            if total_cost == np.inf:
                continue
            _add_candidate(
                new_path,
                total_cost,
                paths,
                num_found,
                cand_paths,
                cand_costs,
                cand_used,
            )

        best_idx, best_cost = _pop_best_candidate(cand_paths, cand_costs, cand_used)
        if best_idx == -1 or best_cost == np.inf:
            break
        for j in range(n_nodes):
            paths[num_found, j] = cand_paths[best_idx, j]
        costs[num_found] = best_cost
        cand_used[best_idx] = False
        cand_costs[best_idx] = np.inf
        num_found += 1
        if num_found >= k_paths:
            break

    return costs, paths
