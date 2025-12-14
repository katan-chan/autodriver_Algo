"""Quick test of hard capacity algorithm with minimal data."""

import numpy as np
from src.algorithms.hard_capacity import solve_routing_hard_capacity_greedy_regret

print("Testing hard capacity algorithm...")

# Create minimal test case
n_nodes = 5
n_vehicles = 3

# Simple linear graph: 0 -> 1 -> 2 -> 3 -> 4
adjacency_travel_time = np.full((n_nodes, n_nodes), np.inf)
for i in range(n_nodes - 1):
    adjacency_travel_time[i, i+1] = 1.0
    adjacency_travel_time[i+1, i] = 1.0

# Bandwidth = 2 for all edges
adjacency_bandwidth = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes - 1):
    adjacency_bandwidth[i, i+1] = 2.0
    adjacency_bandwidth[i+1, i] = 2.0

# 3 vehicles all going from 0 to 4
vehicle_origin = np.array([0, 0, 0], dtype=np.int64)
vehicle_destination = np.array([4, 4, 4], dtype=np.int64)
vehicle_start_time = np.zeros(3, dtype=np.float64)

print("Running algorithm...")
routes, slots, costs = solve_routing_hard_capacity_greedy_regret(
    adjacency_travel_time=adjacency_travel_time,
    adjacency_bandwidth=adjacency_bandwidth,
    vehicle_origin=vehicle_origin,
    vehicle_destination=vehicle_destination,
    vehicle_start_time=vehicle_start_time,
    k_paths=3,
    max_slots_per_edge=10,
)

print("Routes:")
print(routes)
print("\nTest complete!")
