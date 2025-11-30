"""Time-based edge occupancy utilities re-exported for time_regret package."""

from ..common.time_slots import (
    create_edge_time_slots,
    count_vehicles_at_time,
    count_vehicles_in_interval,
    add_vehicle_to_edge,
    remove_vehicle_from_edge,
    get_max_load_in_interval,
    compute_path_times,
    add_path_to_time_slots,
    compute_time_penalty_for_path,
    build_edge_load_timeline,
    build_all_edges_load_timeline,
)

__all__ = [
    "create_edge_time_slots",
    "count_vehicles_at_time",
    "count_vehicles_in_interval",
    "add_vehicle_to_edge",
    "remove_vehicle_from_edge",
    "get_max_load_in_interval",
    "compute_path_times",
    "add_path_to_time_slots",
    "compute_time_penalty_for_path",
    "build_edge_load_timeline",
    "build_all_edges_load_timeline",
]
