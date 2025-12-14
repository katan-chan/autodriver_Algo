"""Routing algorithms for the traffic simulation toy example."""

from .static_regret.baseline import solve_routing_without_penalty
from .static_regret.greedy_regret import solve_routing_with_penalty_greedy_regret
from .static_regret.all_k_paths import compute_all_k_shortest_paths
from .static_regret.edge_utils import build_edge_list_and_index
from .static_regret.incremental_cost import compute_incremental_cost_for_path
from .static_regret.yen import yen_k_shortest_paths

from .time_regret.greedy_regret_time import solve_routing_with_time_penalty_greedy_regret
from .time_regret.evaluation_time import (
    evaluate_time_based_solution,
    print_time_evaluation_report,
    build_edge_time_slots_from_routes,
    build_load_timeline,
)
from .time_regret.time_slots import (
    create_edge_time_slots,
    count_vehicles_at_time,
    count_vehicles_in_interval,
    add_vehicle_to_edge,
    build_all_edges_load_timeline,
)

from .rank_matching.rank_max_matching import solve_rank_maximal_matching

from .hard_capacity.greedy_regret_hard_capacity import solve_routing_hard_capacity_greedy_regret

from .common.dijkstra import dijkstra_shortest_path
from .common.path_cost import compute_path_travel_cost

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
    "solve_rank_maximal_matching",
    "solve_routing_hard_capacity_greedy_regret",
]
