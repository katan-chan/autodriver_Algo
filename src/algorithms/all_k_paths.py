"""Helpers to compute K-shortest paths for every vehicle pair."""

import numpy as np

from .yen import yen_k_shortest_paths


def compute_all_k_shortest_paths(
    adjacency_travel_time: np.ndarray,
    vehicle_origin: np.ndarray,
    vehicle_destination: np.ndarray,
    k_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute K-shortest simple paths for every vehicle origin-destination pair."""

    n_vehicles = vehicle_origin.shape[0]
    n_nodes = adjacency_travel_time.shape[0]

    base_costs = np.full((n_vehicles, k_paths), np.inf)
    all_paths = np.full((n_vehicles, k_paths, n_nodes), -1, dtype=np.int64)

    for vehicle in range(n_vehicles):
        source = int(vehicle_origin[vehicle])
        target = int(vehicle_destination[vehicle])
        costs_v, paths_v = yen_k_shortest_paths(adjacency_travel_time, source, target, k_paths)
        base_costs[vehicle] = costs_v
        all_paths[vehicle] = paths_v

    return base_costs, all_paths
