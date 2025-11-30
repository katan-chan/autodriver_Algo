"""Helper utilities: path finding, edge lists, time slots."""

import numpy as np
from numba import njit


# =============================================================================
# DIJKSTRA
# =============================================================================

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


# =============================================================================
# PATH COST
# =============================================================================

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


# =============================================================================
# YEN K-SHORTEST PATHS (NUMBA)
# =============================================================================

@njit
def yen_k_shortest_paths(
    adjacency_travel_time: np.ndarray,
    source: int,
    target: int,
    k_paths: int,
) -> tuple:
    """Yen's algorithm for K-shortest paths - Numba version."""
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
def compute_all_k_shortest_paths(
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
        costs_v, paths_v = yen_k_shortest_paths(adjacency_travel_time, s, t, k_paths)
        for kk in range(k_paths):
            base_costs[v, kk] = costs_v[kk]
            for j in range(n_nodes):
                all_paths[v, kk, j] = paths_v[kk, j]

    return base_costs, all_paths


# =============================================================================
# EDGE UTILITIES
# =============================================================================

@njit
def build_edge_list_and_index(adjacency_bandwidth: np.ndarray) -> tuple:
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


# =============================================================================
# TIME SLOTS
# =============================================================================

@njit
def create_edge_time_slots(n_edges: int, max_slots: int) -> np.ndarray:
    """
    Tạo ma trận lưu thời gian vào/ra của xe trên mỗi cạnh.
    
    Shape: (n_edges, max_slots, 2)
    - [:, :, 0] = start_time (thời điểm xe vào cạnh)
    - [:, :, 1] = end_time (thời điểm xe ra cạnh)
    - (-1, -1) = slot trống
    """
    return np.full((n_edges, max_slots, 2), -1.0, dtype=np.float64)


@njit
def count_vehicles_at_time(
    edge_time_slots: np.ndarray,
    edge_idx: int,
    query_time: float,
) -> int:
    """Đếm số xe đang ở trên cạnh tại thời điểm query_time."""
    count = 0
    max_slots = edge_time_slots.shape[1]
    
    for slot in range(max_slots):
        start = edge_time_slots[edge_idx, slot, 0]
        end = edge_time_slots[edge_idx, slot, 1]
        
        if start < 0.0:
            continue
        
        if start <= query_time < end:
            count += 1
    
    return count


@njit
def count_vehicles_in_interval(
    edge_time_slots: np.ndarray,
    edge_idx: int,
    interval_start: float,
    interval_end: float,
) -> int:
    """Đếm số xe có overlap với khoảng [interval_start, interval_end)."""
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
def add_vehicle_to_edge(
    edge_time_slots: np.ndarray,
    edge_idx: int,
    start_time: float,
    end_time: float,
) -> int:
    """Thêm xe vào cạnh với thời gian vào/ra. Returns slot index or -1."""
    max_slots = edge_time_slots.shape[1]
    
    for slot in range(max_slots):
        if edge_time_slots[edge_idx, slot, 0] < 0.0:
            edge_time_slots[edge_idx, slot, 0] = start_time
            edge_time_slots[edge_idx, slot, 1] = end_time
            return slot
    
    return -1


@njit
def add_path_to_time_slots(
    edge_time_slots: np.ndarray,
    path: np.ndarray,
    start_time: float,
    adjacency_travel_time: np.ndarray,
    edge_index_matrix: np.ndarray,
    n_nodes: int,
) -> None:
    """Thêm toàn bộ path vào time slots."""
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
        exit_time = current_time + travel_time

        add_vehicle_to_edge(edge_time_slots, edge_idx, entry_time, exit_time)

        current_time = exit_time


@njit
def build_load_timeline(
    edge_time_slots: np.ndarray,
    time_start: float,
    time_end: float,
    n_samples: int,
) -> tuple:
    """
    Build load timeline for all edges.
    
    Returns:
        times: (n_samples,)
        all_loads: (n_edges, n_samples)
    """
    n_edges = edge_time_slots.shape[0]
    times = np.linspace(time_start, time_end, n_samples)
    all_loads = np.zeros((n_edges, n_samples), dtype=np.int64)
    
    for edge_idx in range(n_edges):
        for i in range(n_samples):
            all_loads[edge_idx, i] = count_vehicles_at_time(
                edge_time_slots, edge_idx, times[i]
            )
    
    return times, all_loads
