"""Unified evaluation metrics for routing solutions."""

import numpy as np

from .edge_utils import build_edge_list_and_index


def evaluate_routing_solution(
    routes: np.ndarray,
    adjacency_travel_time: np.ndarray,
    adjacency_bandwidth: np.ndarray,
    beta_penalty: float = 1.0,
) -> dict:
    """
    Đánh giá nghiệm routing với cost function thống nhất (travel + penalty).
    
    Returns dict với:
    - total_travel_time: tổng thời gian di chuyển thuần
    - total_penalty: tổng penalty do vượt băng thông
    - total_cost: travel_time + penalty
    - n_overloaded_edges: số cạnh bị vượt băng thông
    - overload_details: list[(edge_idx, u, v, load, bandwidth, overflow)] cho các cạnh vượt
    - edge_loads: load trên từng cạnh
    """
    n_vehicles, n_nodes = routes.shape
    
    # Build edge structures
    edge_u, edge_v, edge_bandwidth, edge_index_matrix = build_edge_list_and_index(
        adjacency_bandwidth
    )
    n_edges = edge_u.shape[0]
    edge_loads = np.zeros(n_edges, dtype=np.int64)
    
    # 1. Tính travel time và edge loads
    total_travel_time = 0.0
    for vehicle in range(n_vehicles):
        path = routes[vehicle]
        for i in range(n_nodes - 1):
            u = path[i]
            v = path[i + 1]
            if v < 0:
                break
            
            # Travel time
            travel = adjacency_travel_time[u, v]
            if travel != np.inf:
                total_travel_time += travel
            
            # Edge load
            edge_idx = edge_index_matrix[u, v]
            if edge_idx >= 0:
                edge_loads[edge_idx] += 1
    
    # 2. Tính penalty và thống kê overload
    total_penalty = 0.0
    overload_details = []
    
    for edge_idx in range(n_edges):
        load = edge_loads[edge_idx]
        bandwidth = edge_bandwidth[edge_idx]
        
        if load > bandwidth:
            overflow = load - bandwidth
            overflow_ratio = load / bandwidth - 1.0
            penalty = np.exp(beta_penalty * overflow_ratio)
            total_penalty += penalty
            
            overload_details.append({
                "edge_idx": edge_idx,
                "u": int(edge_u[edge_idx]),
                "v": int(edge_v[edge_idx]),
                "load": int(load),
                "bandwidth": int(bandwidth),
                "overflow": int(overflow),
                "overflow_ratio": overflow_ratio,
                "penalty": penalty,
            })
    
    return {
        "total_travel_time": total_travel_time,
        "total_penalty": total_penalty,
        "total_cost": total_travel_time + total_penalty,
        "n_overloaded_edges": len(overload_details),
        "overload_details": overload_details,
        "edge_loads": edge_loads,
        "n_vehicles": n_vehicles,
        "mean_travel_time": total_travel_time / n_vehicles if n_vehicles > 0 else 0.0,
        "mean_cost": (total_travel_time + total_penalty) / n_vehicles if n_vehicles > 0 else 0.0,
    }


def print_evaluation_report(name: str, result: dict) -> None:
    """In báo cáo đánh giá chi tiết."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Vehicles:           {result['n_vehicles']}")
    print(f"  Total travel time:  {result['total_travel_time']:.2f}")
    print(f"  Total penalty:      {result['total_penalty']:.2f}")
    print(f"  Total cost:         {result['total_cost']:.2f}")
    print(f"  Mean travel time:   {result['mean_travel_time']:.2f}")
    print(f"  Mean cost:          {result['mean_cost']:.2f}")
    print(f"  Overloaded edges:   {result['n_overloaded_edges']}")
    
    if result['overload_details']:
        print(f"\n  Chi tiết các cạnh vượt băng thông:")
        print(f"  {'Edge':<10} {'Load':<8} {'BW':<8} {'Overflow':<10} {'Penalty':<12}")
        print(f"  {'-'*48}")
        for detail in result['overload_details']:
            edge_str = f"({detail['u']},{detail['v']})"
            print(f"  {edge_str:<10} {detail['load']:<8} {detail['bandwidth']:<8} "
                  f"+{detail['overflow']:<9} {detail['penalty']:.4f}")
        
        total_overflow = sum(d['overflow'] for d in result['overload_details'])
        print(f"  {'-'*48}")
        print(f"  {'TOTAL':<10} {'':<8} {'':<8} +{total_overflow:<9} {result['total_penalty']:.4f}")
