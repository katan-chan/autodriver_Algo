"""Plotly-based visualization helpers for the traffic simulation."""

from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _generate_n_colors(num_colors: int) -> list[str]:
    """Generate evenly spaced HEX colors on HSV circle."""

    if num_colors <= 0:
        return []

    colors = []
    hues = np.linspace(0.0, 1.0, num_colors, endpoint=False)
    saturation = 0.6
    value = 0.9

    for hue in hues:
        i = int(hue * 6.0)
        f = hue * 6.0 - i
        p = value * (1.0 - saturation)
        q = value * (1.0 - f * saturation)
        t = value * (1.0 - (1.0 - f) * saturation)
        i %= 6
        if i == 0:
            r, g, b = value, t, p
        elif i == 1:
            r, g, b = q, value, p
        elif i == 2:
            r, g, b = p, value, t
        elif i == 3:
            r, g, b = p, q, value
        elif i == 4:
            r, g, b = t, p, value
        else:
            r, g, b = value, p, q
        R = int(r * 255)
        G = int(g * 255)
        B = int(b * 255)
        colors.append(f"#{R:02x}{G:02x}{B:02x}")

    return colors


def _calc_edge_usage(routes: np.ndarray, edge_u: np.ndarray, edge_v: np.ndarray) -> np.ndarray:
    n_vehicles = routes.shape[0]
    usage_map: Dict[tuple[int, int], int] = defaultdict(int)

    for vehicle in range(n_vehicles):
        seq = routes[vehicle]
        seq = seq[seq >= 0]
        if seq.size < 2:
            continue
        for u_node, w_node in zip(seq[:-1], seq[1:]):
            u = int(u_node)
            w = int(w_node)
            if u == w:
                continue
            key = (u, w) if u < w else (w, u)
            usage_map[key] += 1

    usage = np.zeros(edge_u.shape[0], dtype=int)
    for idx, (u, v) in enumerate(zip(edge_u, edge_v)):
        key = (int(u), int(v)) if u < v else (int(v), int(u))
        usage[idx] = usage_map.get(key, 0)

    return usage


def _build_request_graph_traces(
    sample_idx: Iterable[int],
    node_coords: np.ndarray,
    routes: np.ndarray,
    node_size: int,
    req_colors: Dict[int, str],
    req_ids: np.ndarray,
    origins: np.ndarray,
    destinations: np.ndarray,
) -> list[go.Scatter]:
    traces: list[go.Scatter] = []

    for idx in sample_idx:
        rid = int(req_ids[idx])
        origin = int(origins[idx])
        destination = int(destinations[idx])
        color = req_colors[rid]
        group_name = f"req {rid}"

        route_nodes = routes[idx]
        route_nodes = route_nodes[route_nodes >= 0]
        if route_nodes.size >= 2:
            x_route = node_coords[route_nodes, 0]
            y_route = node_coords[route_nodes, 1]
            traces.append(
                go.Scatter(
                    x=x_route,
                    y=y_route,
                    mode="lines",
                    line=dict(width=3, color=color),
                    name=group_name,
                    legendgroup=group_name,
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=[f"{group_name} path node {int(n)}" for n in route_nodes],
                )
            )

        traces.append(
            go.Scatter(
                x=[node_coords[origin, 0]],
                y=[node_coords[origin, 1]],
                mode="markers",
                marker=dict(
                    size=node_size * 1.9,
                    symbol="triangle-up",
                    color=color,
                    line=dict(width=2, color="black"),
                ),
                name=group_name,
                legendgroup=group_name,
                showlegend=True,
                hovertext=[f"{group_name} origin: node {origin} → dest node {destination}"],
                hoverinfo="text",
            )
        )

        traces.append(
            go.Scatter(
                x=[node_coords[destination, 0]],
                y=[node_coords[destination, 1]],
                mode="markers",
                marker=dict(
                    size=node_size * 1.9,
                    symbol="triangle-down",
                    color=color,
                    line=dict(width=2, color="black"),
                ),
                name=group_name,
                legendgroup=group_name,
                showlegend=False,
                hovertext=[f"{group_name} dest: node {destination} (origin {origin})"],
                hoverinfo="text",
            )
        )

    return traces


def visualize_traffic_scenario_plotly_planar(
    data: dict,
    routes: np.ndarray,
    show_vehicle_sample: int | None = None,
    node_size: int = 10,
    title: str = "Traffic Scenario",
    return_fig: bool = False,
) -> go.Figure | None:
    """Visualize planar traffic scenario and per-request routes using Plotly."""

    n_nodes = int(data["n_nodes"])
    node_coords = data["node_coords"]
    edge_u = data["edge_u"]
    edge_v = data["edge_v"]
    edge_bandwidth = data["edge_bandwidth"]
    node_community = data["node_community"]
    origins = data["vehicle_origin"]
    destinations = data["vehicle_destination"]
    start_times = data["vehicle_start_time"]
    req_ids = data["vehicle_id"] + 1
    adj_travel = data["adjacency_travel_time"]
    n_vehicles = int(req_ids.shape[0])

    assert routes.shape[0] == n_vehicles, "routes.shape[0] must equal number of vehicles"

    if show_vehicle_sample is None or show_vehicle_sample >= n_vehicles:
        sample_idx = np.arange(n_vehicles)
    else:
        sample_idx = np.argsort(start_times)[:show_vehicle_sample]

    max_rid = int(req_ids.max())
    req_colors = {rid: color for rid, color in zip(range(1, max_rid + 1), _generate_n_colors(max_rid))}

    edge_usage = _calc_edge_usage(routes, edge_u, edge_v)

    x_nodes = node_coords[:, 0]
    y_nodes = node_coords[:, 1]

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers",
        marker=dict(size=node_size, color="lightgray", line=dict(width=0.5, color="black")),
        text=[f"node {i}, comm {c}" for i, c in enumerate(node_community)],
        hoverinfo="text",
        name="nodes",
    )

    edge_x, edge_y, mid_x, mid_y, edge_text = [], [], [], [], []
    for u_node, v_node, bandwidth, used in zip(edge_u, edge_v, edge_bandwidth, edge_usage):
        x0, y0 = node_coords[u_node]
        x1, y1 = node_coords[v_node]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        mid_x.append((x0 + x1) / 2.0)
        mid_y.append((y0 + y1) / 2.0)
        edge_text.append(f"{int(used)}/{int(bandwidth)}")

    edge_lines_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="lightgray"),
        hoverinfo="none",
        showlegend=False,
    )

    edge_label_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="text",
        text=edge_text,
        textposition="top center",
        textfont=dict(size=10, color="black", family="Arial Black"),
        hoverinfo="text",
        showlegend=False,
        name="usage/capacity",
    )

    request_graph_traces = _build_request_graph_traces(
        sample_idx,
        node_coords,
        routes,
        node_size,
        req_colors,
        req_ids,
        origins,
        destinations,
    )

    timeline_traces: list[go.Bar] = []
    for vehicle_idx, (rid, start) in enumerate(zip(req_ids, start_times)):
        rid_int = int(rid)
        color = req_colors[rid_int]
        group_name = f"req {rid_int}"
        seq = routes[vehicle_idx]
        seq = seq[seq >= 0]
        total_travel = 0.0
        if seq.size >= 2:
            for u_node, w_node in zip(seq[:-1], seq[1:]):
                u = int(u_node)
                w = int(w_node)
                total_travel += float(adj_travel[u, w])
        duration = max(total_travel, 1e-6)

        timeline_traces.append(
            go.Bar(
                x=[duration],
                y=[f"req {rid_int}"],
                base=[start],
                orientation="h",
                marker=dict(color=color),
                name=group_name,
                legendgroup=group_name,
                showlegend=False,
                hovertemplate=(
                    "req %{y}<br>"
                    "start=%{base:.1f}s<br>"
                    "duration=%{x:.1f}s<br>"
                    f"end={start + duration:.1f}s"
                    "<extra></extra>"
                ),
            )
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=("Planar traffic graph", "Request Gantt chart (travel time)"),
    )

    fig.add_trace(edge_lines_trace, row=1, col=1)
    fig.add_trace(edge_label_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    for trace in request_graph_traces:
        fig.add_trace(trace, row=1, col=1)

    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

    for trace in timeline_traces:
        fig.add_trace(trace, row=1, col=2)

    fig.update_xaxes(title_text="time (s)", row=1, col=2)
    fig.update_yaxes(title_text="request", row=1, col=2, autorange="reversed")

    fig.update_layout(
        title=title,
        barmode="stack",
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02),
        height=650,
    )

    if return_fig:
        return fig
    fig.show(renderer="browser")


def visualize_routes_with_time_slider(
    data: dict,
    routes: np.ndarray,
    edge_time_slots: np.ndarray,
    edge_u_arr: np.ndarray,
    edge_v_arr: np.ndarray,
    edge_bandwidth_arr: np.ndarray,
    time_min: float,
    time_max: float,
    n_time_steps: int = 20,
    node_size: int = 10,
    title: str = "Routes with Time Slider",
    return_fig: bool = False,
) -> go.Figure | None:
    """
    Giống visualize gốc nhưng có dropdown chọn thời điểm để xem load/bandwidth.
    Giữ nguyên routes, điểm đầu/cuối, Gantt chart.
    """
    n_nodes = int(data["n_nodes"])
    node_coords = data["node_coords"]
    edge_u = data["edge_u"]
    edge_v = data["edge_v"]
    edge_bandwidth = data["edge_bandwidth"]
    node_community = data["node_community"]
    origins = data["vehicle_origin"]
    destinations = data["vehicle_destination"]
    start_times = data["vehicle_start_time"]
    req_ids = data["vehicle_id"] + 1
    adj_travel = data["adjacency_travel_time"]
    n_vehicles = int(req_ids.shape[0])
    n_edges = len(edge_u_arr)
    max_slots = edge_time_slots.shape[1]

    # Precompute loads at each time step
    time_points = np.linspace(time_min, time_max, n_time_steps)
    loads_over_time = np.zeros((n_edges, n_time_steps), dtype=int)
    
    for t_idx, t in enumerate(time_points):
        for e_idx in range(n_edges):
            count = 0
            for slot in range(max_slots):
                start = edge_time_slots[e_idx, slot, 0]
                end = edge_time_slots[e_idx, slot, 1]
                if start < 0:
                    continue
                if start <= t < end:
                    count += 1
            loads_over_time[e_idx, t_idx] = count

    # Colors and routes
    max_rid = int(req_ids.max())
    req_colors = {rid: color for rid, color in zip(range(1, max_rid + 1), _generate_n_colors(max_rid))}

    x_nodes = node_coords[:, 0]
    y_nodes = node_coords[:, 1]

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers",
        marker=dict(size=node_size, color="lightgray", line=dict(width=0.5, color="black")),
        text=[f"node {i}, comm {c}" for i, c in enumerate(node_community)],
        hoverinfo="text",
        name="nodes",
    )

    # Edge lines (static)
    edge_x, edge_y, mid_x, mid_y = [], [], [], []
    for u_node, v_node in zip(edge_u, edge_v):
        x0, y0 = node_coords[u_node]
        x1, y1 = node_coords[v_node]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        mid_x.append((x0 + x1) / 2.0)
        mid_y.append((y0 + y1) / 2.0)

    edge_lines_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="lightgray"),
        hoverinfo="none",
        showlegend=False,
    )

    # Request traces (routes + origins/destinations)
    request_graph_traces = _build_request_graph_traces(
        np.arange(n_vehicles),
        node_coords,
        routes,
        node_size,
        req_colors,
        req_ids,
        origins,
        destinations,
    )

    # Timeline (Gantt chart)
    timeline_traces: list[go.Bar] = []
    for vehicle_idx, (rid, start) in enumerate(zip(req_ids, start_times)):
        rid_int = int(rid)
        color = req_colors[rid_int]
        group_name = f"req {rid_int}"
        seq = routes[vehicle_idx]
        seq = seq[seq >= 0]
        total_travel = 0.0
        if seq.size >= 2:
            for u_node, w_node in zip(seq[:-1], seq[1:]):
                total_travel += float(adj_travel[int(u_node), int(w_node)])
        duration = max(total_travel, 1e-6)

        timeline_traces.append(
            go.Bar(
                x=[duration],
                y=[f"req {rid_int}"],
                base=[start],
                orientation="h",
                marker=dict(color=color),
                name=group_name,
                legendgroup=group_name,
                showlegend=False,
                hovertemplate=f"req {rid_int}<br>start={start:.1f}s<br>dur={duration:.1f}s<extra></extra>",
            )
        )

    # Build figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=("Network (chọn thời điểm bên dưới)", "Request Gantt chart"),
    )

    # Add static traces
    fig.add_trace(edge_lines_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    for trace in request_graph_traces:
        fig.add_trace(trace, row=1, col=1)
    for trace in timeline_traces:
        fig.add_trace(trace, row=1, col=2)

    # Edge labels - one trace per time step, visibility controlled by dropdown
    label_traces_start_idx = len(fig.data)
    
    for t_idx, t in enumerate(time_points):
        edge_text = []
        text_colors = []
        for e_idx in range(n_edges):
            load = loads_over_time[e_idx, t_idx]
            bw = int(edge_bandwidth_arr[e_idx])
            edge_text.append(f"{load}/{bw}")
            # Color based on load ratio
            ratio = load / bw if bw > 0 else 0
            if ratio <= 0.5:
                text_colors.append("green")
            elif ratio <= 1.0:
                text_colors.append("orange")
            else:
                text_colors.append("red")
        
        fig.add_trace(go.Scatter(
            x=mid_x,
            y=mid_y,
            mode="text",
            text=edge_text,
            textposition="top center",
            textfont=dict(size=9, color=text_colors, family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
            visible=(t_idx == 0),  # Only first visible initially
        ), row=1, col=1)

    # Create slider steps
    n_label_traces = n_time_steps
    slider_steps = []
    for t_idx, t in enumerate(time_points):
        # Create visibility array
        visibility = [True] * label_traces_start_idx  # Keep static traces visible
        visibility += [i == t_idx for i in range(n_label_traces)]  # Show only this time's labels
        
        slider_steps.append(dict(
            args=[{"visible": visibility}],
            label=f"{t:.0f}",
            method="update",
        ))

    fig.update_layout(
        title=title,
        sliders=[{
            "active": 0,
            "currentvalue": {
                "prefix": "Thời điểm: ",
                "suffix": " giây",
                "visible": True,
                "xanchor": "center",
            },
            "pad": {"b": 10, "t": 30},
            "len": 0.55,
            "x": 0.02,
            "y": 0,
            "steps": slider_steps,
        }],
        height=750,
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02),
    )

    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="time (s)", row=1, col=2)
    fig.update_yaxes(title_text="request", row=1, col=2, autorange="reversed")

    if return_fig:
        return fig
    fig.show(renderer="browser")


def visualize_edge_load_timeline(
    times: np.ndarray,
    all_loads: np.ndarray,
    edge_bandwidth: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    top_k_edges: int = 10,
    only_overloaded: bool = True,
    title: str = "Edge Load Over Time",
    return_fig: bool = False,
) -> go.Figure | None:
    """
    Visualize edge load over time as a heatmap + line chart for top edges.
    
    Args:
        times: (n_samples,) time points
        all_loads: (n_edges, n_samples) load at each time
        edge_bandwidth: (n_edges,) bandwidth per edge
        edge_u, edge_v: edge endpoints
        top_k_edges: number of top edges to show in detail
        only_overloaded: if True, only show edges that exceed bandwidth
        title: chart title
    """
    n_edges = all_loads.shape[0]
    
    # Find edges with highest max load
    max_loads = np.max(all_loads, axis=1)
    
    if only_overloaded:
        # Only show edges where max load > bandwidth
        overloaded_mask = max_loads > edge_bandwidth
        overloaded_indices = np.where(overloaded_mask)[0]
        
        if len(overloaded_indices) == 0:
            print(f"  Không có cạnh nào vượt bandwidth!")
            return
        
        # Sort by overflow ratio
        overflow_ratios = max_loads[overloaded_indices] / edge_bandwidth[overloaded_indices]
        sorted_order = np.argsort(overflow_ratios)[::-1]
        top_indices = overloaded_indices[sorted_order][:top_k_edges]
        
        title = f"{title} (Chỉ {len(top_indices)}/{len(overloaded_indices)} cạnh quá tải)"
    else:
        top_indices = np.argsort(max_loads)[::-1][:top_k_edges]
    
    # Nếu nhiều cạnh, chỉ hiện heatmap (line chart sẽ quá lộn xộn)
    show_line_chart = len(top_indices) <= 15
    
    if show_line_chart:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.5, 0.5],
            subplot_titles=("Edge Load Heatmap", "Load vs Time"),
            vertical_spacing=0.12,
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=("Edge Load Heatmap (tất cả cạnh)",),
        )
    
    # Heatmap for top edges
    edge_labels = [f"({edge_u[i]},{edge_v[i]})" for i in top_indices]
    heatmap_data = all_loads[top_indices, :]
    bw_data = edge_bandwidth[top_indices]
    
    # Normalize by bandwidth for color
    heatmap_normalized = np.zeros_like(heatmap_data, dtype=float)
    for i, bw in enumerate(bw_data):
        heatmap_normalized[i, :] = heatmap_data[i, :] / bw
    
    fig.add_trace(
        go.Heatmap(
            z=heatmap_normalized,
            x=times,
            y=edge_labels,
            colorscale=[
                [0.0, "green"],
                [0.33, "yellow"],
                [0.67, "red"],
                [1.0, "darkred"],
            ],
            zmin=0,
            zmax=2.0,
            colorbar=dict(title="Load/BW", y=0.8, len=0.35),
            hovertemplate="Edge: %{y}<br>Time: %{x:.1f}s<br>Load/BW: %{z:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    
    # Line chart for each top edge (only if not too many)
    if show_line_chart:
        colors = _generate_n_colors(len(top_indices))
        
        for idx, edge_idx in enumerate(top_indices):
            u, v = edge_u[edge_idx], edge_v[edge_idx]
            bw = edge_bandwidth[edge_idx]
            loads = all_loads[edge_idx, :]
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=loads,
                    mode="lines",
                    name=f"({u},{v}) BW={int(bw)}",
                    line=dict(color=colors[idx], width=2),
                    hovertemplate=f"Edge ({u},{v})<br>Time: %{{x:.1f}}s<br>Load: %{{y}}<br>BW: {int(bw)}<extra></extra>",
                ),
                row=2, col=1,
            )
            
            # Add bandwidth line (dashed)
            fig.add_trace(
                go.Scatter(
                    x=[times[0], times[-1]],
                    y=[bw, bw],
                    mode="lines",
                    name=f"BW ({u},{v})",
                    line=dict(color=colors[idx], width=1, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2, col=1,
            )
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Load", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Edge", row=1, col=1)
    
    # Adjust height based on number of edges
    n_shown = len(top_indices)
    if show_line_chart:
        chart_height = max(600, 200 + n_shown * 30)  # heatmap + line chart
    else:
        chart_height = max(400, 100 + n_shown * 20)  # heatmap only
    
    layout_opts = dict(title=title, height=chart_height)
    if show_line_chart:
        layout_opts["legend"] = dict(
            orientation="v",
            yanchor="top",
            y=0.45,
            xanchor="left",
            x=1.02,
        )
    
    fig.update_layout(**layout_opts)
    
    if return_fig:
        return fig
    fig.show(renderer="browser")


def visualize_overload_summary(
    times: np.ndarray,
    all_loads: np.ndarray,
    edge_bandwidth: np.ndarray,
    title: str = "Tổng quan quá tải theo thời gian",
    return_fig: bool = False,
) -> go.Figure | None:
    """
    Visualize tổng số cạnh quá tải và tổng overflow theo thời gian.
    Dễ đọc hơn so với heatmap.
    """
    n_edges, n_samples = all_loads.shape
    
    # Tính số cạnh quá tải tại mỗi thời điểm
    n_overloaded = np.zeros(n_samples, dtype=int)
    total_overflow = np.zeros(n_samples, dtype=float)
    
    for t in range(n_samples):
        for e in range(n_edges):
            load = all_loads[e, t]
            bw = edge_bandwidth[e]
            if load > bw:
                n_overloaded[t] += 1
                total_overflow[t] += (load - bw)
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Số cạnh quá tải theo thời gian",
            "Tổng lượng vượt bandwidth theo thời gian"
        ),
        vertical_spacing=0.15,
    )
    
    # Plot 1: Number of overloaded edges
    fig.add_trace(
        go.Scatter(
            x=times,
            y=n_overloaded,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.3)",
            line=dict(color="red", width=2),
            name="Số cạnh quá tải",
        ),
        row=1, col=1,
    )
    
    # Plot 2: Total overflow
    fig.add_trace(
        go.Scatter(
            x=times,
            y=total_overflow,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(255, 165, 0, 0.3)",
            line=dict(color="orange", width=2),
            name="Tổng overflow",
        ),
        row=2, col=1,
    )
    
    fig.update_xaxes(title_text="Thời gian (s)", row=1, col=1)
    fig.update_xaxes(title_text="Thời gian (s)", row=2, col=1)
    fig.update_yaxes(title_text="Số cạnh", row=1, col=1)
    fig.update_yaxes(title_text="Tổng vượt (xe)", row=2, col=1)
    
    # Stats
    max_overloaded = np.max(n_overloaded)
    max_overflow = np.max(total_overflow)
    avg_overloaded = np.mean(n_overloaded)
    
    fig.update_layout(
        title=f"{title}<br><sub>Max {max_overloaded} cạnh quá tải cùng lúc, trung bình {avg_overloaded:.1f} cạnh</sub>",
        height=500,
        showlegend=False,
    )
    
    if return_fig:
        return fig
    fig.show(renderer="browser")


def visualize_edge_gantt_chart(
    edge_time_slots: np.ndarray,
    edge_bandwidth: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    top_k_edges: int = 10,
    title: str = "Edge Occupancy Gantt Chart",
    return_fig: bool = False,
) -> go.Figure | None:
    """
    Visualize vehicle occupancy on edges as Gantt chart.
    
    Each row is an edge, each bar is a vehicle's time on that edge.
    """
    n_edges = edge_time_slots.shape[0]
    max_slots = edge_time_slots.shape[1]
    
    # Count vehicles per edge to find busiest edges
    vehicle_counts = np.zeros(n_edges, dtype=int)
    for e in range(n_edges):
        for s in range(max_slots):
            if edge_time_slots[e, s, 0] >= 0:
                vehicle_counts[e] += 1
    
    top_indices = np.argsort(vehicle_counts)[::-1][:top_k_edges]
    
    fig = go.Figure()
    
    colors = _generate_n_colors(max_slots)
    
    for row_idx, edge_idx in enumerate(top_indices):
        u, v = edge_u[edge_idx], edge_v[edge_idx]
        bw = int(edge_bandwidth[edge_idx])
        edge_label = f"({u},{v}) BW={bw}"
        
        for slot in range(max_slots):
            start = edge_time_slots[edge_idx, slot, 0]
            end = edge_time_slots[edge_idx, slot, 1]
            
            if start < 0:
                continue
            
            duration = end - start
            
            fig.add_trace(
                go.Bar(
                    x=[duration],
                    y=[edge_label],
                    base=[start],
                    orientation="h",
                    marker=dict(color=colors[slot % len(colors)], opacity=0.7),
                    showlegend=False,
                    hovertemplate=(
                        f"Edge ({u},{v})<br>"
                        f"Start: %{{base:.1f}}s<br>"
                        f"End: {end:.1f}s<br>"
                        f"Duration: {duration:.1f}s"
                        "<extra></extra>"
                    ),
                )
            )
    
    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title="Time (s)",
        yaxis_title="Edge",
        height=max(400, top_k_edges * 40),
        yaxis=dict(autorange="reversed"),
    )
    
    if return_fig:
        return fig
    fig.show(renderer="browser")
