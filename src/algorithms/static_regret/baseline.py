"""Baseline solver where each vehicle takes the single shortest path."""

import numpy as np
from numba import njit

from ..common.dijkstra import dijkstra_shortest_path


@njit
def solve_routing_without_penalty(
    adjacency_travel_time: np.ndarray,
    vehicle_origin: np.ndarray,
    vehicle_destination: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each vehicle to its shortest path ignoring bandwidth constraints."""

    n_nodes = adjacency_travel_time.shape[0]
    n_vehicles = vehicle_origin.shape[0]

    routes = np.full((n_vehicles, n_nodes), -1, dtype=np.int64)
    route_costs = np.full(n_vehicles, np.inf)

    for vehicle in range(n_vehicles):
        source = int(vehicle_origin[vehicle])
        target = int(vehicle_destination[vehicle])
        cost, path = dijkstra_shortest_path(adjacency_travel_time, source, target)
        route_costs[vehicle] = cost
        for node_idx in range(n_nodes):
            routes[vehicle, node_idx] = path[node_idx]

    return routes, route_costs
