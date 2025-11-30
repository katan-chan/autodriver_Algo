"""Routing algorithms for the traffic simulation toy example."""

from .baseline import solve_routing_without_penalty
from .greedy_regret import solve_routing_with_penalty_greedy_regret
from .dijkstra import dijkstra_shortest_path
from .yen import yen_k_shortest_paths
from .all_k_paths import compute_all_k_shortest_paths
from .edge_utils import build_edge_list_and_index
from .incremental_cost import compute_incremental_cost_for_path
from .path_cost import compute_path_travel_cost
from .evaluation import evaluate_routing_solution, print_evaluation_report
from .greedy_regret_time import solve_routing_with_time_penalty_greedy_regret
from .evaluation_time import (
    evaluate_time_based_solution,
    print_time_evaluation_report,
    build_edge_time_slots_from_routes,
    build_load_timeline,
)
from .time_slots import (
    create_edge_time_slots,
    count_vehicles_at_time,
    count_vehicles_in_interval,
    add_vehicle_to_edge,
    build_all_edges_load_timeline,
)

__all__ = [
    "solve_routing_without_penalty",
    "solve_routing_with_penalty_greedy_regret",
    "dijkstra_shortest_path",
    "yen_k_shortest_paths",
    "compute_all_k_shortest_paths",
    "build_edge_list_and_index",
    "compute_incremental_cost_for_path",
    "compute_path_travel_cost",
    "evaluate_routing_solution",
    "print_evaluation_report",
    "solve_routing_with_time_penalty_greedy_regret",
    "evaluate_time_based_solution",
    "print_time_evaluation_report",
    "build_edge_time_slots_from_routes",
    "build_load_timeline",
    "create_edge_time_slots",
    "count_vehicles_at_time",
    "count_vehicles_in_interval",
    "add_vehicle_to_edge",
    "build_all_edges_load_timeline",
]
