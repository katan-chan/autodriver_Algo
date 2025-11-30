"""Time-based edge occupancy tracking - shared utilities."""

import numpy as np
from numba import njit


@njit
def create_edge_time_slots(n_edges: int, max_slots: int) -> np.ndarray:
    """Create edge time-slot tensor initialized to -1."""
    return np.full((n_edges, max_slots, 2), -1.0, dtype=np.float64)


@njit
def count_vehicles_at_time(edge_time_slots: np.ndarray, edge_idx: int, query_time: float) -> int:
    """Count vehicles occupying an edge at a specific time."""
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
    """Count vehicles whose occupancy overlaps a given interval."""
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
    """Insert a new occupancy interval for an edge."""
    max_slots = edge_time_slots.shape[1]
    for slot in range(max_slots):
        if edge_time_slots[edge_idx, slot, 0] < 0.0:
            edge_time_slots[edge_idx, slot, 0] = start_time
            edge_time_slots[edge_idx, slot, 1] = end_time
            return slot
    return -1


@njit
def remove_vehicle_from_edge(edge_time_slots: np.ndarray, edge_idx: int, slot_idx: int) -> None:
    """Clear a slot from an edge occupancy tensor."""
    edge_time_slots[edge_idx, slot_idx, 0] = -1.0
    edge_time_slots[edge_idx, slot_idx, 1] = -1.0


@njit
def get_max_load_in_interval(
    edge_time_slots: np.ndarray,
    edge_idx: int,
    interval_start: float,
    interval_end: float,
    time_step: float,
) -> int:
    """Sample an interval and return max load."""
    max_load = 0
    t = interval_start
    while t < interval_end:
        load = count_vehicles_at_time(edge_time_slots, edge_idx, t)
        if load > max_load:
            max_load = load
        t += time_step
    return max_load


@njit
def compute_path_times(
    path: np.ndarray,
    start_time: float,
    adjacency_travel_time: np.ndarray,
    edge_index_matrix: np.ndarray,
    n_nodes: int,
) -> tuple:
    """Compute entry/exit times for each edge on a path."""
    max_path_len = n_nodes - 1
    edge_indices = np.full(max_path_len, -1, dtype=np.int64)
    entry_times = np.full(max_path_len, -1.0, dtype=np.float64)
    exit_times = np.full(max_path_len, -1.0, dtype=np.float64)
    current_time = start_time
    edge_count = 0
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
        edge_indices[edge_count] = edge_idx
        entry_times[edge_count] = current_time
        exit_times[edge_count] = current_time + travel_time
        current_time += travel_time
        edge_count += 1
    return edge_indices, entry_times, exit_times, edge_count


@njit
def add_path_to_time_slots(
    edge_time_slots: np.ndarray,
    path: np.ndarray,
    start_time: float,
    adjacency_travel_time: np.ndarray,
    edge_index_matrix: np.ndarray,
    n_nodes: int,
) -> None:
    """Add every edge traversal of a path into edge_time_slots."""
    edge_indices, entry_times, exit_times, n_edges = compute_path_times(
        path, start_time, adjacency_travel_time, edge_index_matrix, n_nodes
    )
    for i in range(n_edges):
        edge_idx = edge_indices[i]
        if edge_idx >= 0:
            add_vehicle_to_edge(edge_time_slots, edge_idx, entry_times[i], exit_times[i])


@njit
def compute_time_penalty_for_path(
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
    """Compute cost (travel + time-based penalty) if another vehicle uses the path."""

    if base_cost == np.inf:
        return np.inf

    edge_indices, entry_times, exit_times, n_edges = compute_path_times(
        path, start_time, adjacency_travel_time, edge_index_matrix, n_nodes
    )

    total_penalty = 0.0

    for i in range(n_edges):
        edge_idx = edge_indices[i]
        if edge_idx < 0:
            continue

        entry_time = entry_times[i]
        exit_time = exit_times[i]

        current_load = count_vehicles_in_interval(
            edge_time_slots, edge_idx, entry_time, exit_time
        )

        bandwidth = edge_bandwidth[edge_idx]
        new_load = float(current_load) + 1.0
        overflow_ratio = new_load / bandwidth - 1.0

        if overflow_ratio > 0.0:
            total_penalty += np.exp(beta_penalty * overflow_ratio)

    return base_cost + total_penalty


@njit
def build_edge_load_timeline(
    edge_time_slots: np.ndarray,
    edge_idx: int,
    time_start: float,
    time_end: float,
    n_samples: int,
) -> tuple:
    """Build sampled load timeline for a single edge."""

    times = np.linspace(time_start, time_end, n_samples)
    loads = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        loads[i] = count_vehicles_at_time(edge_time_slots, edge_idx, times[i])

    return times, loads


@njit
def build_all_edges_load_timeline(
    edge_time_slots: np.ndarray,
    time_start: float,
    time_end: float,
    n_samples: int,
) -> tuple:
    """Return sampled loads for every edge over time."""
    n_edges = edge_time_slots.shape[0]
    times = np.linspace(time_start, time_end, n_samples)
    all_loads = np.zeros((n_edges, n_samples), dtype=np.int64)
    for edge_idx in range(n_edges):
        for i in range(n_samples):
            all_loads[edge_idx, i] = count_vehicles_at_time(edge_time_slots, edge_idx, times[i])
    return times, all_loads
