"""Greedy + regret assignment using Yen K paths and congestion penalty."""

import numpy as np

from .all_k_paths import compute_all_k_shortest_paths
from .edge_utils import build_edge_list_and_index
from .incremental_cost import compute_incremental_cost_for_path


def solve_routing_with_penalty_greedy_regret(
    adjacency_travel_time: np.ndarray,
    adjacency_bandwidth: np.ndarray,
    vehicle_origin: np.ndarray,
    vehicle_destination: np.ndarray,
    k_paths: int = 3,
    beta_penalty: float = 1.0,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign vehicles greedily with regret heuristic using congestion penalty."""

    n_nodes = adjacency_travel_time.shape[0]
    n_vehicles = vehicle_origin.shape[0]

    base_costs, all_paths = compute_all_k_shortest_paths(
        adjacency_travel_time,
        vehicle_origin,
        vehicle_destination,
        k_paths,
    )

    edge_u, edge_v, edge_bandwidth, edge_index_matrix = build_edge_list_and_index(adjacency_bandwidth)
    n_edges = edge_u.shape[0]

    edge_loads = np.zeros(n_edges, dtype=np.int64)
    routes_final = np.full((n_vehicles, n_nodes), -1, dtype=np.int64)
    assigned = np.zeros(n_vehicles, dtype=np.bool_)
    remaining = n_vehicles

    T1 = np.zeros(n_vehicles, dtype=np.float64)
    T2 = np.zeros(n_vehicles, dtype=np.float64)
    regret = np.zeros(n_vehicles, dtype=np.float64)
    best_k_for_v = np.zeros(n_vehicles, dtype=np.int64)

    BIG_REGRET = 1e12
    iteration = 0

    while remaining > 0:
        T1.fill(np.inf)
        T2.fill(np.inf)
        regret.fill(-1.0)
        best_k_for_v.fill(-1)

        for vehicle in range(n_vehicles):
            if assigned[vehicle]:
                continue
            for k_idx in range(k_paths):
                base_cost = base_costs[vehicle, k_idx]
                if base_cost == np.inf:
                    continue
                path = all_paths[vehicle, k_idx]
                total_cost = compute_incremental_cost_for_path(
                    path,
                    base_cost,
                    edge_index_matrix,
                    edge_bandwidth,
                    edge_loads,
                    beta_penalty,
                )
                if total_cost < T1[vehicle]:
                    T2[vehicle] = T1[vehicle]
                    T1[vehicle] = total_cost
                    best_k_for_v[vehicle] = k_idx
                elif total_cost < T2[vehicle]:
                    T2[vehicle] = total_cost

            if T1[vehicle] < np.inf:
                regret[vehicle] = BIG_REGRET if T2[vehicle] == np.inf else T2[vehicle] - T1[vehicle]

        if debug:
            print(f"\n===== ITERATION {iteration} =====")
            print("v | assigned | best_k |     T1     |     T2     |      regret")
            for vehicle in range(n_vehicles):
                if assigned[vehicle]:
                    continue
                if best_k_for_v[vehicle] == -1:
                    print(f"{vehicle:2d} |    1?    |   -1   |       inf |       inf |         -  ")
                else:
                    print(
                        f"{vehicle:2d} |    0     | {best_k_for_v[vehicle]:5d} | "
                        f"{T1[vehicle]:9.2f} | {T2[vehicle]:9.2f} | {regret[vehicle]:11.2f}"
                    )

        best_vehicle = -1
        best_regret_value = -1.0
        best_T1_value = np.inf
        for vehicle in range(n_vehicles):
            if assigned[vehicle] or best_k_for_v[vehicle] == -1:
                continue
            r = regret[vehicle]
            t1 = T1[vehicle]
            if r > best_regret_value or (r == best_regret_value and t1 < best_T1_value):
                best_regret_value = r
                best_T1_value = t1
                best_vehicle = vehicle

        if best_vehicle == -1:
            if debug:
                print("No more feasible vehicles, stopping.")
            break

        k_star = best_k_for_v[best_vehicle]
        chosen_path = all_paths[best_vehicle, k_star]
        routes_final[best_vehicle] = chosen_path

        for idx in range(n_nodes - 1):
            u = chosen_path[idx]
            w = chosen_path[idx + 1]
            if w < 0:
                break
            edge_idx = edge_index_matrix[u, w]
            if edge_idx >= 0:
                edge_loads[edge_idx] += 1

        assigned[best_vehicle] = True
        remaining -= 1
        iteration += 1

    return routes_final, edge_loads, base_costs
