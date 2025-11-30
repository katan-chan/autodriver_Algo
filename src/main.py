"""Main entry point for the traffic routing simulation."""

import os
import numpy as np

from .fake_data import generate_planar_traffic_data
from .visualize import (
    visualize_traffic_scenario_plotly_planar,
    visualize_routes_with_time_slider,
    visualize_edge_load_timeline,
    visualize_edge_gantt_chart,
    visualize_overload_summary,
)
from .visualize_html import export_figures_to_tabbed_html
from .algorithms import (
    solve_routing_without_penalty,
    solve_routing_with_time_penalty_greedy_regret,
    evaluate_time_based_solution,
    print_time_evaluation_report,
    build_load_timeline,
)


def main() -> None:
    # ============================================================
    # 1. Sinh dữ liệu giả lập
    # ============================================================
    print("=== Generating Data ===")
    data = generate_planar_traffic_data(
        n_nodes=90,
        n_vehicles=90,
        n_communities=3,
        p_in=0.7,
        p_out=0.5,
        bandwidth_low=5,
        bandwidth_high=6,
        seed=42,
    )
    
    # Parameters
    k_paths = 10
    beta_penalty = 1
    max_slots_per_edge = 200  # max vehicles per edge slot

    # ============================================================
    # 2. Baseline: Shortest path, không xét băng thông
    # ============================================================
    print("\n=== Running Baseline: Shortest Path ===")
    routes_baseline, _ = solve_routing_without_penalty(
        adjacency_travel_time=data["adjacency_travel_time"],
        vehicle_origin=data["vehicle_origin"],
        vehicle_destination=data["vehicle_destination"],
    )

    # ============================================================
    # 3. Thuật toán chính: Greedy + Regret với TIME-BASED bandwidth
    # ============================================================
    print("\n=== Running Greedy + Regret (Time-Based, K=10) ===")
    routes_time, edge_time_slots, base_costs = solve_routing_with_time_penalty_greedy_regret(
        adjacency_travel_time=data["adjacency_travel_time"],
        adjacency_bandwidth=data["adjacency_bandwidth"],
        vehicle_origin=data["vehicle_origin"],
        vehicle_destination=data["vehicle_destination"],
        vehicle_start_time=data["vehicle_start_time"],
        k_paths=k_paths,
        beta_penalty=beta_penalty,
        max_slots_per_edge=max_slots_per_edge,
    )

    # ============================================================
    # 4. Đánh giá với TIME-BASED cost function
    # ============================================================
    print("\n=== Evaluating Solutions ===")
    time_sample_step = 5.0  # sample mỗi 5 giây
    
    eval_baseline = evaluate_time_based_solution(
        routes_baseline,
        data["vehicle_start_time"],
        data["adjacency_travel_time"],
        data["adjacency_bandwidth"],
        beta_penalty,
        max_slots_per_edge,
        time_sample_step,
    )
    
    eval_time = evaluate_time_based_solution(
        routes_time,
        data["vehicle_start_time"],
        data["adjacency_travel_time"],
        data["adjacency_bandwidth"],
        beta_penalty,
        max_slots_per_edge,
        time_sample_step,
    )
    
    print_time_evaluation_report("Baseline: Shortest Path", eval_baseline)
    print_time_evaluation_report("Greedy + Regret (Time-Based)", eval_time)
    
    # So sánh nhanh
    print("\n" + "=" * 60)
    print("  SO SÁNH")
    print("=" * 60)
    cost_diff = eval_baseline[2] - eval_time[2]  # total_cost
    cost_diff_pct = cost_diff / eval_baseline[2] * 100 if eval_baseline[2] > 0 else 0
    print(f"  Greedy+Regret tiết kiệm: {cost_diff:.2f} ({cost_diff_pct:.1f}% tổng cost)")
    print(f"  Baseline: {eval_baseline[3]} overloaded intervals, max overflow = {eval_baseline[4]}")
    print(f"  Time-Based: {eval_time[3]} overloaded intervals, max overflow = {eval_time[4]}")

    # ============================================================
    # 5. Build all visualizations
    # ============================================================
    print("\n=== Building Visualizations ===")
    figures = []
    
    # Get time range
    time_min = float(np.min(data["vehicle_start_time"]))
    time_max = time_min + 600.0  # 10 minutes window
    n_samples = 200
    
    # Build timeline data
    times_baseline, loads_baseline = build_load_timeline(
        eval_baseline[5], time_min, time_max, n_samples,
    )
    times_time, loads_time = build_load_timeline(
        eval_time[5], time_min, time_max, n_samples,
    )
    n_edges = len(eval_baseline[7])
    
    # 1. Routes with Time Dropdown - Baseline
    print("  Building: Routes Baseline (dropdown thời gian)")
    fig = visualize_routes_with_time_slider(
        data,
        routes_baseline,
        eval_baseline[5],  # edge_time_slots
        eval_baseline[7],  # edge_u
        eval_baseline[8],  # edge_v
        eval_baseline[9],  # edge_bandwidth
        time_min, time_max,
        n_time_steps=20,
        node_size=12,
        title="Baseline: Routes + Load/BW (chọn thời điểm)",
        return_fig=True,
    )
    figures.append(("1. Routes Baseline", fig))
    
    # 2. Routes with Time Dropdown - Greedy
    print("  Building: Routes Greedy+Regret (dropdown thời gian)")
    fig = visualize_routes_with_time_slider(
        data,
        routes_time,
        eval_time[5],  # edge_time_slots
        eval_time[7],  # edge_u
        eval_time[8],  # edge_v
        eval_time[9],  # edge_bandwidth
        time_min, time_max,
        n_time_steps=20,
        node_size=12,
        title="Greedy+Regret: Routes + Load/BW (chọn thời điểm)",
        return_fig=True,
    )
    figures.append(("2. Routes Greedy", fig))
    
    # 3. Overload Summary - Baseline
    print("  Building: Overload Summary Baseline")
    fig = visualize_overload_summary(
        times_baseline, loads_baseline, eval_baseline[9],
        title="Baseline: Tổng quan quá tải",
        return_fig=True,
    )
    figures.append(("3. Overload Baseline", fig))
    
    # 4. Overload Summary - Greedy
    print("  Building: Overload Summary Greedy")
    fig = visualize_overload_summary(
        times_time, loads_time, eval_time[9],
        title="Greedy + Regret: Tổng quan quá tải",
        return_fig=True,
    )
    figures.append(("4. Overload Greedy", fig))
    
    # 5. Heatmap all edges - Baseline
    print(f"  Building: Heatmap {n_edges} edges Baseline")
    fig = visualize_edge_load_timeline(
        times_baseline, loads_baseline, eval_baseline[9],
        eval_baseline[7], eval_baseline[8],
        top_k_edges=n_edges, only_overloaded=False,
        title=f"Baseline: Tất cả {n_edges} cạnh",
        return_fig=True,
    )
    figures.append(("5. Heatmap Baseline", fig))
    
    # 6. Heatmap all edges - Greedy
    print(f"  Building: Heatmap {n_edges} edges Greedy")
    fig = visualize_edge_load_timeline(
        times_time, loads_time, eval_time[9],
        eval_time[7], eval_time[8],
        top_k_edges=n_edges, only_overloaded=False,
        title=f"Greedy + Regret: Tất cả {n_edges} cạnh",
        return_fig=True,
    )
    figures.append(("6. Heatmap Greedy", fig))
    
    # 7. Gantt - Baseline
    print("  Building: Gantt Baseline")
    fig = visualize_edge_gantt_chart(
        eval_baseline[5], eval_baseline[9],
        eval_baseline[7], eval_baseline[8],
        top_k_edges=15,
        title="Baseline: Edge Occupancy Gantt",
        return_fig=True,
    )
    figures.append(("7. Gantt Baseline", fig))
    
    # 8. Gantt - Greedy
    print("  Building: Gantt Greedy")
    fig = visualize_edge_gantt_chart(
        eval_time[5], eval_time[9],
        eval_time[7], eval_time[8],
        top_k_edges=15,
        title="Greedy + Regret: Edge Occupancy Gantt",
        return_fig=True,
    )
    figures.append(("8. Gantt Greedy", fig))
    
    # ============================================================
    # 6. Export to single HTML with tabs
    # ============================================================
    output_path = os.path.join(os.path.dirname(__file__), "..", "results.html")
    output_path = os.path.abspath(output_path)
    
    export_figures_to_tabbed_html(
        figures,
        output_path,
        title="Traffic Simulation: Baseline vs Greedy+Regret",
    )


if __name__ == "__main__":
    main()
