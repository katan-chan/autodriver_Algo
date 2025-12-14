"""Main entry point for hard capacity constraint routing simulation."""

import os
import numpy as np

from .fake_data import generate_planar_traffic_data
from .visualize import (
    visualize_overload_summary,
    build_algorithm_dropdown_figures,
    visualize_edge_load_timeline,
)
from .visualize_html import export_figures_to_tabbed_html
from .algorithms.hard_capacity import solve_routing_hard_capacity_greedy_regret
from .algorithms import (
    solve_routing_without_penalty,
    evaluate_time_based_solution,
    print_time_evaluation_report,
    build_load_timeline,
)


def main() -> None:
    # ============================================================
    # 1. Sinh dữ liệu giả lập
    # ============================================================
    print("=== Generating Data (Hard Capacity Version) ===")
    data = generate_planar_traffic_data(
        n_nodes=90,
        n_vehicles=90,
        n_communities=3,
        p_in=0.7,
        p_out=1,
        bandwidth_low=3,
        bandwidth_high=5,
        seed=42,
        uniform_travel_time=False,  # Travel time ngẫu nhiên theo khoảng cách
        uniform_start_time=True,    # Tất cả xe xuất phát tại t = 0
    )
    
    # ============================================================
    # 2. Parameters
    # ============================================================
    k_paths = 20  # Reduced from 10 for faster testing
    max_slots_per_edge = 200

    # ============================================================
    # 3. Baseline: Shortest path, không xét băng thông
    # ============================================================
    print("\n=== Running Baseline: Shortest Path ===")
    routes_baseline, _ = solve_routing_without_penalty(
        adjacency_travel_time=data["adjacency_travel_time"],
        vehicle_origin=data["vehicle_origin"],
        vehicle_destination=data["vehicle_destination"],
    )

    # ============================================================
    # 4. Simple Greedy: Hard Capacity (no regret)
    # ============================================================
    print("\n=== Running Simple Greedy (Hard Capacity, K=20) ===")
    from .algorithms.hard_capacity import solve_routing_hard_capacity_simple_greedy
    
    routes_simple, edge_time_slots_simple, base_costs_simple = solve_routing_hard_capacity_simple_greedy(
        adjacency_travel_time=data["adjacency_travel_time"],
        adjacency_bandwidth=data["adjacency_bandwidth"],
        vehicle_origin=data["vehicle_origin"],
        vehicle_destination=data["vehicle_destination"],
        vehicle_start_time=data["vehicle_start_time"],
        k_paths=k_paths,
        max_slots_per_edge=max_slots_per_edge,
    )

    # ============================================================
    # 5. Thuật toán mới: Hard Capacity Greedy Regret
    # ============================================================
    print("\n=== Running Greedy + Regret (Hard Capacity, K=20) ===")
    routes_hard, edge_time_slots_hard, base_costs_hard = solve_routing_hard_capacity_greedy_regret(
        adjacency_travel_time=data["adjacency_travel_time"],
        adjacency_bandwidth=data["adjacency_bandwidth"],
        vehicle_origin=data["vehicle_origin"],
        vehicle_destination=data["vehicle_destination"],
        vehicle_start_time=data["vehicle_start_time"],
        k_paths=k_paths,
        max_slots_per_edge=max_slots_per_edge,
    )

    # ============================================================
    # 6. Đánh giá solutions
    # ============================================================
    print("\n=== Evaluating Solutions ===")
    time_sample_step = 5.0  # sample mỗi 5 giây
    
    # Evaluate baseline (không có penalty vì chỉ tính travel cost)
    eval_baseline = evaluate_time_based_solution(
        routes_baseline,
        data["vehicle_start_time"],
        data["adjacency_travel_time"],
        data["adjacency_bandwidth"],
        beta_penalty=0.0,  # No penalty for baseline
        max_slots_per_edge=max_slots_per_edge,
        time_sample_step=time_sample_step,
    )
    
    # Evaluate simple greedy
    eval_simple = evaluate_time_based_solution(
        routes_simple,
        data["vehicle_start_time"],
        data["adjacency_travel_time"],
        data["adjacency_bandwidth"],
        beta_penalty=0.0,
        max_slots_per_edge=max_slots_per_edge,
        time_sample_step=time_sample_step,
    )
    
    # Evaluate hard capacity (greedy + regret)
    eval_hard = evaluate_time_based_solution(
        routes_hard,
        data["vehicle_start_time"],
        data["adjacency_travel_time"],
        data["adjacency_bandwidth"],
        beta_penalty=0.0,  # No penalty, just check violations
        max_slots_per_edge=max_slots_per_edge,
        time_sample_step=time_sample_step,
    )
    
    print_time_evaluation_report("Baseline: Shortest Path", eval_baseline)
    print_time_evaluation_report("Simple Greedy (Hard Capacity)", eval_simple)
    print_time_evaluation_report("Greedy+Regret (Hard Capacity)", eval_hard)
    
    # Count assigned vehicles
    n_assigned_baseline = np.sum([1 for r in routes_baseline if r[0] >= 0])
    n_assigned_simple = np.sum([1 for r in routes_simple if r[0] >= 0])
    n_assigned_hard = np.sum([1 for r in routes_hard if r[0] >= 0])
    
    # So sánh
    print("\n" + "=" * 60)
    print("  SO SÁNH")
    print("=" * 60)
    print(f"  Baseline: {n_assigned_baseline}/{len(routes_baseline)} vehicles assigned")
    print(f"  Simple Greedy: {n_assigned_simple}/{len(routes_simple)} vehicles assigned")
    print(f"  Greedy+Regret: {n_assigned_hard}/{len(routes_hard)} vehicles assigned")
    print(f"  Baseline: {eval_baseline[3]} overloaded intervals, max overflow = {eval_baseline[4]}")
    print(f"  Simple Greedy: {eval_simple[3]} overloaded intervals, max overflow = {eval_simple[4]}")
    print(f"  Greedy+Regret: {eval_hard[3]} overloaded intervals, max overflow = {eval_hard[4]}")
    
    if eval_simple[4] == 0:
        print("\n  ✓ Simple Greedy: No capacity violations!")
    if eval_hard[4] == 0:
        print("  ✓ Greedy+Regret: No capacity violations!")

    # ============================================================
    # 6. Build visualizations
    # ============================================================
    print("\n=== Building Visualizations ===")
    figures = []
    
    # Get time range
    time_min = 0.0  # All vehicles start at t=0
    time_max = 600.0  # 10 phút window
    n_samples = 200
    
    # Build timeline data
    times_baseline, loads_baseline = build_load_timeline(
        eval_baseline[5], time_min, time_max, n_samples,
    )
    times_simple, loads_simple = build_load_timeline(
        eval_simple[5], time_min, time_max, n_samples,
    )
    times_hard, loads_hard = build_load_timeline(
        eval_hard[5], time_min, time_max, n_samples,
    )

    solution_payloads = {
        "Baseline": {
            "routes": routes_baseline,
            "edge_time_slots": eval_baseline[5],
            "edge_u": eval_baseline[7],
            "edge_v": eval_baseline[8],
            "edge_bandwidth": eval_baseline[9],
            "times": times_baseline,
            "loads": loads_baseline,
        },
        "Simple-Greedy": {
            "routes": routes_simple,
            "edge_time_slots": eval_simple[5],
            "edge_u": eval_simple[7],
            "edge_v": eval_simple[8],
            "edge_bandwidth": eval_simple[9],
            "times": times_simple,
            "loads": loads_simple,
        },
        "Greedy+Regret": {
            "routes": routes_hard,
            "edge_time_slots": eval_hard[5],
            "edge_u": eval_hard[7],
            "edge_v": eval_hard[8],
            "edge_bandwidth": eval_hard[9],
            "times": times_hard,
            "loads": loads_hard,
        },
    }

    print("  Building: Routes dropdown figures")
    routes_dropdown = build_algorithm_dropdown_figures(
        data=data,
        solutions=solution_payloads,
        time_min=time_min,
        time_max=time_max,
        n_time_steps=20,
        node_size=12,
    )

    print("  Building: Overload summary dropdown")
    overload_dropdown = []
    for name, payload in solution_payloads.items():
        fig = visualize_overload_summary(
            payload["times"], payload["loads"], payload["edge_bandwidth"],
            title=f"{name}: Tổng quan quá tải",
            return_fig=True,
        )
        overload_dropdown.append((name, fig))

    print("  Building: Heatmap dropdown")
    heatmap_dropdown = []
    for name, payload in solution_payloads.items():
        fig = visualize_edge_load_timeline(
            payload["times"],
            payload["loads"],
            payload["edge_bandwidth"],
            payload["edge_u"],
            payload["edge_v"],
            top_k_edges=20,
            only_overloaded=False,
            title=f"{name}: Heatmap + Load Timeline",
            return_fig=True,
            show_line_chart=True,
        )
        heatmap_dropdown.append((name, fig))

    figures = [
        ("Routes + Load/BW", {"type": "dropdown", "options": routes_dropdown}),
        ("Overload Summary", {"type": "dropdown", "options": overload_dropdown}),
        ("Heatmap + Timeline", {"type": "dropdown", "options": heatmap_dropdown}),
    ]
    
    # ============================================================
    # 7. Export to HTML
    # ============================================================
    output_path = os.path.join(os.path.dirname(__file__), "..", "results_hard_capacity.html")
    output_path = os.path.abspath(output_path)
    
    export_figures_to_tabbed_html(
        figures,
        output_path,
        title="Hard Capacity Routing: Baseline vs Simple-Greedy vs Greedy+Regret",
    )
    
    print(f"\n=== Complete! Results saved to: {output_path} ===")


if __name__ == "__main__":
    main()
