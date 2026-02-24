"""
Graph topology analysis utilities.

This module provides functions for analyzing graph structure,
selecting hub nodes, and manipulating graph topology.
"""

import torch
import numpy as np
from typing import Tuple, List
from torch_geometric.data import Data


def get_node_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute node degrees.

    Args:
        edge_index: Graph connectivity [2, num_edges]
        num_nodes: Total number of nodes

    Returns:
        Node degrees [num_nodes]

    Example:
        >>> degrees = get_node_degrees(data.edge_index, data.num_nodes)
        >>> high_degree_nodes = (degrees > 10).nonzero()
    """
    row, col = edge_index
    degrees = torch.bincount(row, minlength=num_nodes)
    return degrees


def select_high_degree_nodes(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 100
) -> torch.Tensor:
    """
    Select top-k high-degree nodes.

    Args:
        edge_index: Graph connectivity [2, num_edges]
        num_nodes: Total number of nodes
        k: Number of nodes to select

    Returns:
        Indices of top-k high-degree nodes [k]

    Example:
        >>> hub_candidates = select_high_degree_nodes(data.edge_index, data.num_nodes, k=50)
    """
    degrees = get_node_degrees(edge_index, num_nodes)
    _, top_indices = torch.topk(degrees, min(k, num_nodes))
    return top_indices


def get_neighbors(edge_index: torch.Tensor, node_idx: int) -> torch.Tensor:
    """
    Get all neighbors of a node.

    Args:
        edge_index: Graph connectivity [2, num_edges]
        node_idx: Target node index

    Returns:
        Neighbor indices [num_neighbors]

    Example:
        >>> neighbors = get_neighbors(data.edge_index, hub_idx)
        >>> target_idx = neighbors[0].item()
    """
    row, col = edge_index
    mask = row == node_idx
    neighbors = col[mask]
    return neighbors


def get_k_hop_neighbors(
    edge_index: torch.Tensor,
    node_idx: int,
    k: int = 2,
    num_nodes: int = None
) -> List[torch.Tensor]:
    """
    Get k-hop neighbors of a node.

    Args:
        edge_index: Graph connectivity [2, num_edges]
        node_idx: Target node index
        k: Number of hops
        num_nodes: Total number of nodes (optional)

    Returns:
        List of neighbor sets for each hop [k]
        Each element is a tensor of node indices

    Example:
        >>> hop_neighbors = get_k_hop_neighbors(data.edge_index, target_idx, k=3)
        >>> l1_neighbors = hop_neighbors[0]
        >>> l2_neighbors = hop_neighbors[1]
    """
    row, col = edge_index

    current_nodes = torch.tensor([node_idx])
    visited = set([node_idx])
    hop_neighbors = []

    for _ in range(k):
        # Find all neighbors of current nodes
        next_nodes = []
        for node in current_nodes:
            neighbors = col[row == node]
            for neighbor in neighbors:
                neighbor_idx = neighbor.item()
                if neighbor_idx not in visited:
                    next_nodes.append(neighbor_idx)
                    visited.add(neighbor_idx)

        if len(next_nodes) == 0:
            break

        next_nodes = torch.tensor(list(set(next_nodes)))
        hop_neighbors.append(next_nodes)
        current_nodes = next_nodes

    return hop_neighbors


def select_control_node(
    edge_index: torch.Tensor,
    num_nodes: int,
    hub_idx: int,
    target_idx: int,
    seed: int = None
) -> int:
    """
    Select a control node (non-neighbor of hub and target).

    Args:
        edge_index: Graph connectivity [2, num_edges]
        num_nodes: Total number of nodes
        hub_idx: Hub node index
        target_idx: Target node index
        seed: Random seed for selection (optional)

    Returns:
        Control node index

    Example:
        >>> control_idx = select_control_node(data.edge_index, data.num_nodes,
        ...                                    hub_idx, target_idx, seed=42)
    """
    # Get neighbors of hub and target
    hub_neighbors = get_neighbors(edge_index, hub_idx)
    target_neighbors = get_neighbors(edge_index, target_idx)

    # Combine excluded nodes
    excluded = set([hub_idx, target_idx])
    excluded.update(hub_neighbors.tolist())
    excluded.update(target_neighbors.tolist())

    # Find valid control candidates
    all_nodes = set(range(num_nodes))
    candidates = list(all_nodes - excluded)

    if len(candidates) == 0:
        raise ValueError("No valid control nodes available")

    # Select control node
    if seed is not None:
        np.random.seed(seed)

    control_idx = np.random.choice(candidates)
    return control_idx


def remove_node_edges(data: Data, node_idx: int) -> Data:
    """
    Remove all edges connected to a node (for unlearning).

    Args:
        data: PyTorch Geometric data object
        node_idx: Node to remove edges from

    Returns:
        New data object with edges removed

    Example:
        >>> data_unlearned = remove_node_edges(data, target_idx)
    """
    row, col = data.edge_index
    mask = (row != node_idx) & (col != node_idx)

    new_data = data.clone()
    new_data.edge_index = data.edge_index[:, mask]

    return new_data


def filter_edges_by_nodes_vectorized(
    edge_index: torch.Tensor,
    node_indices: torch.Tensor,
    device: torch.device = None
) -> torch.Tensor:
    """
    Vectorized edge filtering - O(E) complexity instead of O(E×N).

    Filters edges to keep only those where BOTH endpoints are in node_indices.

    Args:
        edge_index: Graph connectivity [2, num_edges]
        node_indices: Node indices to keep [num_selected_nodes]
        device: Device for computation (optional)

    Returns:
        Boolean mask [num_edges] indicating which edges to keep

    Example:
        >>> shard_nodes = torch.tensor([0, 1, 5, 10, 15])
        >>> mask = filter_edges_by_nodes_vectorized(data.edge_index, shard_nodes)
        >>> filtered_edges = data.edge_index[:, mask]
    """
    if device is None:
        device = edge_index.device

    row, col = edge_index
    # Include node_indices in num_nodes calculation to handle isolated nodes
    max_edge = max(row.max().item(), col.max().item()) if edge_index.numel() > 0 else -1
    max_node = node_indices.max().item() if node_indices.numel() > 0 else -1
    num_nodes = max(max_edge, max_node) + 1

    # Create a boolean mask for nodes in the set
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    node_mask[node_indices] = True

    # Vectorized check: both endpoints must be in node_indices
    row_in = node_mask[row]
    col_in = node_mask[col]
    edge_mask = row_in & col_in

    return edge_mask


def remap_edge_indices(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    node_indices: torch.Tensor,
    device: torch.device = None
) -> torch.Tensor:
    """
    Remap edge indices to local indices after filtering.

    Args:
        edge_index: Original edge_index [2, num_edges]
        edge_mask: Boolean mask from filter_edges_by_nodes_vectorized
        node_indices: Node indices (sorted) [num_selected_nodes]
        device: Device for computation

    Returns:
        Remapped edge_index [2, num_filtered_edges] with local indices

    Example:
        >>> mask = filter_edges_by_nodes_vectorized(edge_index, shard_nodes)
        >>> local_edges = remap_edge_indices(edge_index, mask, shard_nodes)
    """
    if device is None:
        device = edge_index.device

    # Get filtered edges
    filtered_edges = edge_index[:, edge_mask]

    if filtered_edges.size(1) == 0:
        return torch.tensor([[], []], dtype=torch.long, device=device)

    # Create mapping from global to local indices
    num_nodes = max(edge_index.max().item(), node_indices.max().item()) + 1
    global_to_local = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    global_to_local[node_indices] = torch.arange(len(node_indices), device=device)

    # Remap
    local_row = global_to_local[filtered_edges[0]]
    local_col = global_to_local[filtered_edges[1]]

    return torch.stack([local_row, local_col], dim=0)


def get_k_hop_neighbors_fast(
    edge_index: torch.Tensor,
    node_idx: int,
    k: int = 2,
    num_nodes: int = None
) -> List[torch.Tensor]:
    """
    Optimized k-hop neighbor search using sparse matrix operations.

    Args:
        edge_index: Graph connectivity [2, num_edges]
        node_idx: Target node index
        k: Number of hops
        num_nodes: Total number of nodes

    Returns:
        List of neighbor tensors for each hop distance [k]

    Example:
        >>> hop_neighbors = get_k_hop_neighbors_fast(data.edge_index, target_idx, k=3)
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    device = edge_index.device
    row, col = edge_index

    # Build adjacency list using scatter
    # This is more efficient than repeated row == node comparisons
    adj_list = [[] for _ in range(num_nodes)]
    for r, c in zip(row.tolist(), col.tolist()):
        adj_list[r].append(c)

    visited = {node_idx}
    current_frontier = {node_idx}
    hop_neighbors = []

    for _ in range(k):
        next_frontier = set()
        for node in current_frontier:
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)

        if not next_frontier:
            break

        hop_neighbors.append(torch.tensor(list(next_frontier), device=device))
        current_frontier = next_frontier

    return hop_neighbors


def compute_shortest_path_length(
    edge_index: torch.Tensor,
    num_nodes: int,
    source: int,
    target: int,
    max_hops: int = 10
) -> int:
    """
    Compute shortest path length between two nodes (BFS).

    Args:
        edge_index: Graph connectivity [2, num_edges]
        num_nodes: Total number of nodes
        source: Source node index
        target: Target node index
        max_hops: Maximum hops to search

    Returns:
        Shortest path length (-1 if not reachable)

    Example:
        >>> dist = compute_shortest_path_length(data.edge_index, data.num_nodes,
        ...                                       hub_idx, target_idx)
    """
    if source == target:
        return 0

    row, col = edge_index
    visited = {source}
    queue = [(source, 0)]

    while queue:
        node, dist = queue.pop(0)

        if dist >= max_hops:
            break

        # Get neighbors
        neighbors = col[row == node]

        for neighbor in neighbors:
            neighbor_idx = neighbor.item()

            if neighbor_idx == target:
                return dist + 1

            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                queue.append((neighbor_idx, dist + 1))

    return -1  # Not reachable
