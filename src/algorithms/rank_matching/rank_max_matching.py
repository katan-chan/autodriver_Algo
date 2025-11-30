"""Rank-maximal matching heuristic for the Autodriver routing problem."""

from __future__ import annotations

import numpy as np
from numba import njit

from ..static_regret.all_k_paths import compute_all_k_shortest_paths
from ..static_regret.edge_utils import build_edge_list_and_index
from ..static_regret.incremental_cost import compute_incremental_cost_for_path


@njit
def _apply_path_load(edge_loads: np.ndarray, path: np.ndarray, edge_index_matrix: np.ndarray, delta: int) -> None:
    n_nodes = path.shape[0]
    for idx in range(n_nodes - 1):
        u = path[idx]
        v = path[idx + 1]
        if v < 0:
            break
        edge_idx = edge_index_matrix[u, v]
        if edge_idx >= 0:
            edge_loads[edge_idx] += delta


@njit
def _paths_share_edge(path_a: np.ndarray, path_b: np.ndarray) -> bool:
    n_nodes = path_a.shape[0]
    for idx in range(n_nodes - 1):
        u = path_a[idx]
        v = path_a[idx + 1]
        if v < 0:
            break
        for jdx in range(n_nodes - 1):
            x = path_b[jdx]
            y = path_b[jdx + 1]
            if y < 0:
                break
            if (u == x and v == y) or (u == y and v == x):
                return True
    return False


def solve_rank_maximal_matching(
    adjacency_travel_time: np.ndarray,
    adjacency_bandwidth: np.ndarray,
    vehicle_origin: np.ndarray,
    vehicle_destination: np.ndarray,
    k_paths: int = 3,
    beta_penalty: float = 1.0,
    direct_accept_margin: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implement the rank-maximal heuristic described in matching_algo.md."""

    n_nodes = adjacency_travel_time.shape[0]
    n_vehicles = vehicle_origin.shape[0]

    base_costs, all_paths = compute_all_k_shortest_paths(
        adjacency_travel_time,
        vehicle_origin,
        vehicle_destination,
        k_paths,
    )

    edge_u, edge_v, edge_bandwidth, edge_index_matrix = build_edge_list_and_index(adjacency_bandwidth)
    n_edges = edge_u.shape[0]

    edge_loads = np.zeros(n_edges, dtype=np.int64)
    routes_final = np.full((n_vehicles, n_nodes), -1, dtype=np.int64)
    assigned_path_index = np.full(n_vehicles, -1, dtype=np.int64)
    current_costs = np.full(n_vehicles, np.inf)

    vehicle_order = np.argsort(-base_costs[:, 0])
    epsilon = direct_accept_margin

    for vehicle in vehicle_order:
        best_path = all_paths[vehicle, 0]
        base_cost = base_costs[vehicle, 0]
        direct_cost = compute_incremental_cost_for_path(
            best_path,
            base_cost,
            edge_index_matrix,
            edge_bandwidth,
            edge_loads,
            beta_penalty,
        )

        if direct_cost - base_cost <= epsilon:
            _apply_path_load(edge_loads, best_path, edge_index_matrix, +1)
            routes_final[vehicle] = best_path
            assigned_path_index[vehicle] = 0
            current_costs[vehicle] = direct_cost
            continue

        best_option_type = "A"
        best_option_value = np.inf
        best_option_data = None

        for alt_idx in range(1, k_paths):
            alt_path = all_paths[vehicle, alt_idx]
            if alt_path[0] < 0:
                break
            alt_cost = compute_incremental_cost_for_path(
                alt_path,
                base_costs[vehicle, alt_idx],
                edge_index_matrix,
                edge_bandwidth,
                edge_loads,
                beta_penalty,
            )
            if alt_cost < best_option_value:
                best_option_value = alt_cost
                best_option_data = (alt_idx, alt_cost)

        for other in range(n_vehicles):
            if assigned_path_index[other] == -1:
                continue
            current_other_k = assigned_path_index[other]
            if current_other_k + 1 >= k_paths:
                continue
            if not _paths_share_edge(routes_final[other], best_path):
                continue

            alt_idx = current_other_k + 1
            alt_path_other = all_paths[other, alt_idx]
            if alt_path_other[0] < 0:
                continue

            _apply_path_load(edge_loads, routes_final[other], edge_index_matrix, -1)

            cost_vehicle = compute_incremental_cost_for_path(
                best_path,
                base_cost,
                edge_index_matrix,
                edge_bandwidth,
                edge_loads,
                beta_penalty,
            )

            cost_other_alt = compute_incremental_cost_for_path(
                alt_path_other,
                base_costs[other, alt_idx],
                edge_index_matrix,
                edge_bandwidth,
                edge_loads,
                beta_penalty,
            )

            delta_total = (cost_vehicle + cost_other_alt) - current_costs[other]

            _apply_path_load(edge_loads, routes_final[other], edge_index_matrix, +1)

            if delta_total < best_option_value:
                best_option_type = "B"
                best_option_value = delta_total
                best_option_data = (other, alt_idx, cost_vehicle, cost_other_alt)

        if best_option_data is None:
            _apply_path_load(edge_loads, best_path, edge_index_matrix, +1)
            routes_final[vehicle] = best_path
            assigned_path_index[vehicle] = 0
            current_costs[vehicle] = direct_cost
            continue

        if best_option_type == "A":
            alt_idx, alt_cost = best_option_data
            alt_path = all_paths[vehicle, alt_idx]
            _apply_path_load(edge_loads, alt_path, edge_index_matrix, +1)
            routes_final[vehicle] = alt_path
            assigned_path_index[vehicle] = alt_idx
            current_costs[vehicle] = alt_cost
        else:
            other_vehicle, new_k, cost_vehicle, cost_other_alt = best_option_data
            _apply_path_load(edge_loads, routes_final[other_vehicle], edge_index_matrix, -1)
            new_path_other = all_paths[other_vehicle, new_k]
            _apply_path_load(edge_loads, new_path_other, edge_index_matrix, +1)
            routes_final[other_vehicle] = new_path_other
            assigned_path_index[other_vehicle] = new_k
            current_costs[other_vehicle] = cost_other_alt

            _apply_path_load(edge_loads, best_path, edge_index_matrix, +1)
            routes_final[vehicle] = best_path
            assigned_path_index[vehicle] = 0
            current_costs[vehicle] = cost_vehicle

    return routes_final, edge_loads, assigned_path_index
