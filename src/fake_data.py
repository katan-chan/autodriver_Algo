"""Synthetic planar traffic data generation utilities."""

from typing import Dict

import networkx as nx
import numpy as np


def generate_planar_traffic_data(
    n_nodes: int = 100,
    n_vehicles: int = 30,
    n_communities: int = 4,
    p_in: float = 0.9,
    p_out: float = 0.7,
    bandwidth_low: int = 5,
    bandwidth_high: int = 20,
    time_factor_low: float = 30.0,
    time_factor_high: float = 90.0,
    seed: int = 42,
    time_window_seconds: float = 300.0,
) -> Dict[str, np.ndarray]:
    """Generate synthetic planar traffic data with per-edge bandwidth and travel time."""

    rng = np.random.default_rng(seed)

    # =============== 1. Sinh đồ thị phẳng cơ bản ===============
    rows = int(np.floor(np.sqrt(n_nodes)))
    cols = int(np.ceil(n_nodes / rows))
    total_nodes = rows * cols

    G_grid = nx.grid_2d_graph(rows, cols)  # planar

    mapping = {}
    reverse_mapping = {}
    node_id = 0
    for i in range(rows):
        for j in range(cols):
            mapping[(i, j)] = node_id
            reverse_mapping[node_id] = (i, j)
            node_id += 1

    G = nx.Graph()
    for nid in range(min(total_nodes, n_nodes)):
        G.add_node(nid)

    for (u2, v2) in G_grid.edges():
        u = mapping[u2]
        v = mapping[v2]
        if u < n_nodes and v < n_nodes:
            G.add_edge(u, v)

    n_nodes = G.number_of_nodes()

    # =============== 2. Cộng đồng + xoá cạnh ===============
    node_community = np.zeros(n_nodes, dtype=int)
    cols_per_comm = max(1, cols // n_communities)
    for nid in range(n_nodes):
        i, j = reverse_mapping[nid]
        comm = min(j // cols_per_comm, n_communities - 1)
        node_community[nid] = comm

    # Lọc cạnh
    to_remove = []
    for u, v in list(G.edges()):
        cu, cv = node_community[u], node_community[v]
        prob = p_in if cu == cv else p_out
        if rng.random() > prob:
            to_remove.append((u, v))
    G.remove_edges_from(to_remove)

    # =============== 3. Đảm bảo liên thông ===============
    def ensure_connected(G_):
        while not nx.is_connected(G_):
            comps = list(nx.connected_components(G_))
            base = comps[0]
            for u in base:
                i, j = reverse_mapping[u]
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ii, jj = i + di, j + dj
                    if (ii, jj) not in mapping:
                        continue
                    v = mapping[(ii, jj)]
                    if v < n_nodes and not G_.has_edge(u, v):
                        G_.add_edge(u, v)
                        break
                if nx.is_connected(G_):
                    break

    ensure_connected(G)

    # =============== 4. Gán toạ độ, B_e, T_e ===============
    node_coords = np.zeros((n_nodes, 2))
    for nid in range(n_nodes):
        i, j = reverse_mapping[nid]
        node_coords[nid] = [j, -i]

    edges = list(G.edges())
    m = len(edges)

    edge_u = np.zeros(m, dtype=int)
    edge_v = np.zeros(m, dtype=int)

    # B_e random nguyên, mỗi cạnh một giá trị
    edge_bandwidth = rng.integers(bandwidth_low, bandwidth_high + 1, size=m).astype(float)

    # factor_e random, mỗi cạnh một giá trị
    edge_time_factor = rng.uniform(time_factor_low, time_factor_high, size=m)
    edge_travel_time = np.zeros(m, dtype=float)

    adj_bandwidth = np.zeros((n_nodes, n_nodes), dtype=float)
    adj_travel = np.full((n_nodes, n_nodes), np.inf, dtype=float)

    for k, (u, v) in enumerate(edges):
        edge_u[k], edge_v[k] = u, v

        # dist trên grid
        du = node_coords[u] - node_coords[v]
        dist = float(np.linalg.norm(du))

        # travel time = dist * factor_e
        t = dist * edge_time_factor[k]
        edge_travel_time[k] = t

        B = edge_bandwidth[k]
        adj_bandwidth[u, v] = adj_bandwidth[v, u] = B
        adj_travel[u, v] = adj_travel[v, u] = t

    # =============== 5. Sinh request (5 phút) ===============
    origins = rng.integers(0, n_nodes, size=n_vehicles)
    destinations = rng.integers(0, n_nodes, size=n_vehicles)
    for i in range(n_vehicles):
        if origins[i] == destinations[i]:
            choices = np.setdiff1d(np.arange(n_nodes), [origins[i]])
            destinations[i] = rng.choice(choices)
    start_times = rng.uniform(0, time_window_seconds, size=n_vehicles)

    # =============== 6. Gói toàn bộ dữ liệu vào 1 dict duy nhất ===============
    data = dict(
        graph=G,
        node_coords=node_coords,
        node_community=node_community,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_bandwidth=edge_bandwidth,
        edge_travel_time=edge_travel_time,
        edge_time_factor=edge_time_factor,
        adjacency_bandwidth=adj_bandwidth,
        adjacency_travel_time=adj_travel,
        vehicle_origin=origins,
        vehicle_destination=destinations,
        vehicle_start_time=start_times,
        vehicle_id=np.arange(n_vehicles),
        n_nodes=n_nodes,
        n_edges=m,
        n_vehicles=n_vehicles,
    )

    return data
