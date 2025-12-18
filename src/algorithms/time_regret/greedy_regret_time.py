"""Greedy + regret assignment with time-based bandwidth tracking - Numba compatible."""

import numpy as np
from numba import njit

from ..common.dijkstra import dijkstra_shortest_path
from ..common.path_cost import compute_path_travel_cost


@njit
def _yen_k_shortest_paths_numba(
    adjacency_travel_time: np.ndarray,
    source: int,
    target: int,
    k_paths: int,
) -> tuple:
    """Yen's algorithm - Numba version."""
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
        for i in range(n_nodes):
            if prev_path[i] == -1:
                break
            path_len += 1

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

            # Check duplicate
            is_dup = False
            for ii in range(num_found):
                same = True
                for jj in range(n_nodes):
                    if paths[ii, jj] != new_path[jj]:
                        same = False
                        break
                    if paths[ii, jj] == -1 and new_path[jj] == -1:
                        break
                if same:
                    is_dup = True
                    break

            if not is_dup:
                for c in range(max_candidates):
                    if cand_used[c]:
                        same = True
                        for jj in range(n_nodes):
                            if cand_paths[c, jj] != new_path[jj]:
                                same = False
                                break
                            if cand_paths[c, jj] == -1 and new_path[jj] == -1:
                                break
                        if same:
                            is_dup = True
                            break

            if not is_dup:
                free_idx = -1
                for c in range(max_candidates):
                    if not cand_used[c]:
                        free_idx = c
                        break
                if free_idx >= 0:
                    for j in range(n_nodes):
                        cand_paths[free_idx, j] = new_path[j]
                    cand_costs[free_idx] = total_cost
                    cand_used[free_idx] = True

        best_idx = -1
        best_cost = np.inf
        for c in range(max_candidates):
            if cand_used[c] and cand_costs[c] < best_cost:
                best_cost = cand_costs[c]
                best_idx = c

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


@njit
def _compute_all_k_paths_numba(
    adjacency_travel_time: np.ndarray,
    vehicle_origin: np.ndarray,
    vehicle_destination: np.ndarray,
    k_paths: int,
) -> tuple:
    """Compute K-shortest paths for all vehicles."""
    n_vehicles = vehicle_origin.shape[0]
    n_nodes = adjacency_travel_time.shape[0]

    base_costs = np.full((n_vehicles, k_paths), np.inf)
    all_paths = np.full((n_vehicles, k_paths, n_nodes), -1, dtype=np.int64)

    for v in range(n_vehicles):
        s = vehicle_origin[v]
        t = vehicle_destination[v]
        costs_v, paths_v = _yen_k_shortest_paths_numba(adjacency_travel_time, s, t, k_paths)
        for kk in range(k_paths):
            base_costs[v, kk] = costs_v[kk]
            for j in range(n_nodes):
                all_paths[v, kk, j] = paths_v[kk, j]

    return base_costs, all_paths


@njit
def _count_vehicles_in_interval(
    edge_time_slots: np.ndarray,
    edge_idx: int,
    interval_start: float,
    interval_end: float,
) -> int:
    """Count vehicles overlapping with interval."""
    count = 0
    max_slots = edge_time_slots.shape[1]

    for slot in range(max_slots):
        start = edge_time_slots[edge_idx, slot, 0]
        end = edge_time_slots[edge_idx, slot, 1]

        if start < 0.0:
            continue

        if start < interval_end and end > interval_start:
            count += 1

    return count


@njit
def _add_vehicle_to_edge(
    edge_time_slots: np.ndarray,
    edge_idx: int,
    start_time: float,
    end_time: float,
) -> int:
    """Add vehicle to edge time slot."""
    max_slots = edge_time_slots.shape[1]

    for slot in range(max_slots):
        if edge_time_slots[edge_idx, slot, 0] < 0.0:
            edge_time_slots[edge_idx, slot, 0] = start_time
            edge_time_slots[edge_idx, slot, 1] = end_time
            return slot

    return -1


@njit
def _compute_time_penalty_for_path(
    path: np.ndarray,
    start_time: float,
    base_cost: float,
    adjacency_travel_time: np.ndarray,
    edge_index_matrix: np.ndarray,
    edge_bandwidth: np.ndarray,
    edge_time_slots: np.ndarray,
    beta_penalty: float,
    n_nodes: int,
) -> float:
    """Compute cost with time-based penalty."""
    if base_cost == np.inf:
        return np.inf

    total_penalty = 0.0
    current_time = start_time

    for i in range(n_nodes - 1):
        u = path[i]
        v = path[i + 1]

        if v < 0:
            break

        edge_idx = edge_index_matrix[u, v]
        if edge_idx < 0:
            continue

        travel_time = adjacency_travel_time[u, v]
        if travel_time == np.inf:
            break

        entry_time = current_time
        
        # Count current load in this time interval (tạm tính với travel_time gốc)
        temp_exit_time = current_time + travel_time
        current_load = _count_vehicles_in_interval(
            edge_time_slots, edge_idx, entry_time, temp_exit_time
        )

        bandwidth = edge_bandwidth[edge_idx]
        new_load = float(current_load) + 1.0
        overflow_ratio = new_load / bandwidth - 1.0

        if overflow_ratio > 0.0:
            # Penalty time: làm chậm xe do tắc đường
            penalty_time = (np.exp(beta_penalty * overflow_ratio) - 1.0)
            total_penalty += penalty_time
            # Exit time bao gồm cả penalty (xe đi chậm hơn)
            exit_time = current_time + travel_time + penalty_time
        else:
            exit_time = current_time + travel_time

        current_time = exit_time

    return base_cost + total_penalty


@njit
def _add_path_to_time_slots(
    edge_time_slots: np.ndarray,
    path: np.ndarray,
    start_time: float,
    adjacency_travel_time: np.ndarray,
    edge_index_matrix: np.ndarray,
    edge_bandwidth: np.ndarray,
    beta_penalty: float,
    n_nodes: int,
) -> None:
    """Add entire path to time slots, accounting for penalty delays."""
    current_time = start_time

    for i in range(n_nodes - 1):
        u = path[i]
        v = path[i + 1]

        if v < 0:
            break

        edge_idx = edge_index_matrix[u, v]
        if edge_idx < 0:
            continue

        travel_time = adjacency_travel_time[u, v]
        if travel_time == np.inf:
            break

        entry_time = current_time
        
        # Tính penalty dựa trên load hiện tại
        temp_exit_time = current_time + travel_time
        current_load = _count_vehicles_in_interval(
            edge_time_slots, edge_idx, entry_time, temp_exit_time
        )
        
        bandwidth = edge_bandwidth[edge_idx]
        new_load = float(current_load) + 1.0
        overflow_ratio = new_load / bandwidth - 1.0
        
        if overflow_ratio > 0.0:
            # Có penalty, xe đi chậm hơn
            penalty_time = travel_time * (np.exp(beta_penalty * overflow_ratio) - 1.0)
            exit_time = current_time + travel_time + penalty_time
        else:
            exit_time = current_time + travel_time

        _add_vehicle_to_edge(edge_time_slots, edge_idx, entry_time, exit_time)

        current_time = exit_time


@njit
def _build_edge_list_numba(adjacency_bandwidth: np.ndarray) -> tuple:
    """Build edge list from adjacency matrix."""
    n_nodes = adjacency_bandwidth.shape[0]

    # Count edges
    m = 0
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if adjacency_bandwidth[u, v] > 0.0:
                m += 1

    edge_u = np.zeros(m, dtype=np.int64)
    edge_v = np.zeros(m, dtype=np.int64)
    edge_bandwidth = np.zeros(m, dtype=np.float64)
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


@njit
def solve_routing_with_time_penalty_greedy_regret(
    adjacency_travel_time: np.ndarray,
    adjacency_bandwidth: np.ndarray,
    vehicle_origin: np.ndarray,
    vehicle_destination: np.ndarray,
    vehicle_start_time: np.ndarray,
    k_paths: int,
    beta_penalty: float,
    max_slots_per_edge: int,
) -> tuple:
    """
    Greedy + Regret với time-based bandwidth tracking.

    Args:
        adjacency_travel_time: (n_nodes, n_nodes) travel time matrix
        adjacency_bandwidth: (n_nodes, n_nodes) bandwidth matrix
        vehicle_origin: (n_vehicles,) origin nodes
        vehicle_destination: (n_vehicles,) destination nodes
        vehicle_start_time: (n_vehicles,) departure times
        k_paths: number of alternative paths per vehicle
        beta_penalty: penalty coefficient
        max_slots_per_edge: max vehicles tracked per edge

    Returns:
        routes_final: (n_vehicles, n_nodes) chosen paths
        edge_time_slots: (n_edges, max_slots, 2) time slot matrix
        base_costs: (n_vehicles, k_paths) base travel costs
    """
    n_nodes = adjacency_travel_time.shape[0]
    n_vehicles = vehicle_origin.shape[0]

    # Build edge structures
    edge_u, edge_v, edge_bandwidth, edge_index_matrix = _build_edge_list_numba(
        adjacency_bandwidth
    )
    n_edges = edge_u.shape[0]

    # Compute K-shortest paths
    base_costs, all_paths = _compute_all_k_paths_numba(
        adjacency_travel_time, vehicle_origin, vehicle_destination, k_paths
    )

    # Initialize time slots
    edge_time_slots = np.full((n_edges, max_slots_per_edge, 2), -1.0, dtype=np.float64)

    # Output arrays
    routes_final = np.full((n_vehicles, n_nodes), -1, dtype=np.int64)
    assigned = np.zeros(n_vehicles, dtype=np.bool_)
    remaining = n_vehicles

    # Temp arrays for each iteration
    T1 = np.zeros(n_vehicles, dtype=np.float64)
    T2 = np.zeros(n_vehicles, dtype=np.float64)
    regret = np.zeros(n_vehicles, dtype=np.float64)
    best_k_for_v = np.zeros(n_vehicles, dtype=np.int64)

    BIG_REGRET = 1e12

    while remaining > 0:
        # Reset
        for v in range(n_vehicles):
            T1[v] = np.inf
            T2[v] = np.inf
            regret[v] = -1.0
            best_k_for_v[v] = -1

        # Compute T1, T2, regret for each unassigned vehicle
        for v in range(n_vehicles):
            if assigned[v]:
                continue

            start_time = vehicle_start_time[v]

            for kk in range(k_paths):
                base_cost = base_costs[v, kk]
                if base_cost == np.inf:
                    continue

                path = all_paths[v, kk]
                total_cost = _compute_time_penalty_for_path(
                    path,
                    start_time,
                    base_cost,
                    adjacency_travel_time,
                    edge_index_matrix,
                    edge_bandwidth,
                    edge_time_slots,
                    beta_penalty,
                    n_nodes,
                )

                if total_cost < T1[v]:
                    T2[v] = T1[v]
                    T1[v] = total_cost
                    best_k_for_v[v] = kk
                elif total_cost < T2[v]:
                    T2[v] = total_cost

            if T1[v] < np.inf:
                if T2[v] < np.inf:
                    regret[v] = T2[v] - T1[v]
                else:
                    regret[v] = BIG_REGRET

        # Find vehicle with max regret
        best_v = -1
        best_regret = -1.0
        best_T1 = np.inf

        for v in range(n_vehicles):
            if assigned[v] or best_k_for_v[v] == -1:
                continue

            r = regret[v]
            t1 = T1[v]

            if r > best_regret or (r == best_regret and t1 < best_T1):
                best_regret = r
                best_T1 = t1
                best_v = v

        if best_v == -1:
            break

        # Assign best path to best vehicle
        k_star = best_k_for_v[best_v]
        chosen_path = all_paths[best_v, k_star]
        start_time = vehicle_start_time[best_v]

        for i in range(n_nodes):
            routes_final[best_v, i] = chosen_path[i]

        # Add path to time slots
        _add_path_to_time_slots(
            edge_time_slots,
            chosen_path,
            start_time,
            adjacency_travel_time,
            edge_index_matrix,
            edge_bandwidth,
            beta_penalty,
            n_nodes,
        )

        assigned[best_v] = True
        remaining -= 1

    return routes_final, edge_time_slots, base_costs
