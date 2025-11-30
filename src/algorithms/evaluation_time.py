"""Time-based evaluation metrics for routing solutions - Numba compatible."""

import numpy as np
from numba import njit


@njit
def _build_edge_list_numba(adjacency_bandwidth: np.ndarray) -> tuple:
    """Build edge list from adjacency matrix."""
    n_nodes = adjacency_bandwidth.shape[0]

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
def _count_vehicles_at_time(
    edge_time_slots: np.ndarray,
    edge_idx: int,
    query_time: float,
) -> int:
    """Count vehicles at specific time."""
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
def build_edge_time_slots_from_routes(
    routes: np.ndarray,
    vehicle_start_time: np.ndarray,
    adjacency_travel_time: np.ndarray,
    adjacency_bandwidth: np.ndarray,
    max_slots_per_edge: int,
) -> tuple:
    """
    Build edge time slots from routes.
    
    Returns:
        edge_time_slots: (n_edges, max_slots, 2)
        edge_u, edge_v, edge_bandwidth, edge_index_matrix
    """
    n_vehicles, n_nodes = routes.shape
    
    edge_u, edge_v, edge_bandwidth, edge_index_matrix = _build_edge_list_numba(
        adjacency_bandwidth
    )
    n_edges = edge_u.shape[0]
    
    edge_time_slots = np.full((n_edges, max_slots_per_edge, 2), -1.0, dtype=np.float64)
    
    for vehicle in range(n_vehicles):
        path = routes[vehicle]
        current_time = vehicle_start_time[vehicle]
        
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
            
            # Find empty slot
            for slot in range(max_slots_per_edge):
                if edge_time_slots[edge_idx, slot, 0] < 0.0:
                    edge_time_slots[edge_idx, slot, 0] = entry_time
                    edge_time_slots[edge_idx, slot, 1] = exit_time
                    break
            
            current_time = exit_time
    
    return edge_time_slots, edge_u, edge_v, edge_bandwidth, edge_index_matrix


@njit
def evaluate_time_based_solution(
    routes: np.ndarray,
    vehicle_start_time: np.ndarray,
    adjacency_travel_time: np.ndarray,
    adjacency_bandwidth: np.ndarray,
    beta_penalty: float,
    max_slots_per_edge: int,
    time_sample_step: float,
) -> tuple:
    """
    Evaluate routing solution with time-based metrics.
    
    Returns tuple of:
        - total_travel_time: float
        - total_penalty: float
        - total_cost: float
        - n_overloaded_intervals: int (number of (edge, time) pairs exceeding bandwidth)
        - max_overload: int (max overflow across all edges and times)
        - edge_time_slots: (n_edges, max_slots, 2)
        - edge_max_loads: (n_edges,) max load per edge
        - edge_u, edge_v, edge_bandwidth
    """
    n_vehicles, n_nodes = routes.shape
    
    # Build time slots
    edge_time_slots, edge_u, edge_v, edge_bandwidth, edge_index_matrix = \
        build_edge_time_slots_from_routes(
            routes, vehicle_start_time, adjacency_travel_time,
            adjacency_bandwidth, max_slots_per_edge
        )
    n_edges = edge_u.shape[0]
    
    # Compute total travel time
    total_travel_time = 0.0
    time_min = np.inf
    time_max = -np.inf
    
    for vehicle in range(n_vehicles):
        path = routes[vehicle]
        current_time = vehicle_start_time[vehicle]
        
        if current_time < time_min:
            time_min = current_time
        
        for i in range(n_nodes - 1):
            u = path[i]
            v = path[i + 1]
            
            if v < 0:
                break
            
            travel_time = adjacency_travel_time[u, v]
            if travel_time != np.inf:
                total_travel_time += travel_time
                current_time += travel_time
        
        if current_time > time_max:
            time_max = current_time
    
    # Compute penalty and overload stats
    total_penalty = 0.0
    n_overloaded_intervals = 0
    max_overload = 0
    edge_max_loads = np.zeros(n_edges, dtype=np.int64)
    
    # Sample time points
    if time_max > time_min:
        n_samples = int((time_max - time_min) / time_sample_step) + 1
    else:
        n_samples = 1
    
    for edge_idx in range(n_edges):
        bandwidth = int(edge_bandwidth[edge_idx])
        edge_max_load = 0
        
        for s in range(n_samples):
            t = time_min + s * time_sample_step
            load = _count_vehicles_at_time(edge_time_slots, edge_idx, t)
            
            if load > edge_max_load:
                edge_max_load = load
            
            if load > bandwidth:
                n_overloaded_intervals += 1
                overflow = load - bandwidth
                if overflow > max_overload:
                    max_overload = overflow
                
                # Penalty: time_step * exp(beta * overflow_ratio)
                overflow_ratio = float(load) / float(bandwidth) - 1.0
                total_penalty += beta_penalty * time_sample_step * np.exp(max(load - bandwidth, 0))
                # total_penalty += beta_penalty * time_sample_step * overflow_ratio
        
        edge_max_loads[edge_idx] = edge_max_load
    
    total_cost = total_travel_time + total_penalty
    
    return (
        total_travel_time,
        total_penalty,
        total_cost,
        n_overloaded_intervals,
        max_overload,
        edge_time_slots,
        edge_max_loads,
        edge_u,
        edge_v,
        edge_bandwidth,
    )


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
            all_loads[edge_idx, i] = _count_vehicles_at_time(
                edge_time_slots, edge_idx, times[i]
            )
    
    return times, all_loads


def print_time_evaluation_report(name: str, result: tuple) -> None:
    """Print time-based evaluation report (non-Numba)."""
    (
        total_travel_time,
        total_penalty,
        total_cost,
        n_overloaded_intervals,
        max_overload,
        edge_time_slots,
        edge_max_loads,
        edge_u,
        edge_v,
        edge_bandwidth,
    ) = result
    
    n_edges = edge_u.shape[0]
    n_vehicles = 0
    for slot in range(edge_time_slots.shape[1]):
        if edge_time_slots[0, slot, 0] >= 0:
            n_vehicles += 1
    
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Total travel time:       {total_travel_time:.2f}")
    print(f"  Total penalty:           {total_penalty:.2f}")
    print(f"  Total cost:              {total_cost:.2f}")
    print(f"  Overloaded intervals:    {n_overloaded_intervals}")
    print(f"  Max overflow:            {max_overload}")
    
    # Find overloaded edges
    overloaded_edges = []
    for e in range(n_edges):
        if edge_max_loads[e] > edge_bandwidth[e]:
            overloaded_edges.append(e)
    
    print(f"  Overloaded edges:        {len(overloaded_edges)}")
    
    if overloaded_edges:
        print(f"\n  Chi tiết các cạnh vượt băng thông:")
        print(f"  {'Edge':<12} {'MaxLoad':<10} {'BW':<8} {'Overflow':<10}")
        print(f"  {'-'*40}")
        
        total_overflow = 0
        for e in overloaded_edges:
            u = edge_u[e]
            v = edge_v[e]
            load = edge_max_loads[e]
            bw = int(edge_bandwidth[e])
            overflow = load - bw
            total_overflow += overflow
            print(f"  ({u},{v}){'':<6} {load:<10} {bw:<8} +{overflow:<9}")
        
        print(f"  {'-'*40}")
        print(f"  {'TOTAL':<12} {'':<10} {'':<8} +{total_overflow}")
