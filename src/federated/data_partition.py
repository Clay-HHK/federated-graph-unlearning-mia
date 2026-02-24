"""
Graph partitioning strategies for federated graph learning.

Supports:
- Metis: Minimizes edge cut (locality preserving, ~5-10% cross-edges)
- Random: Uniform random assignment (structure breaking, ~30-40% cross-edges)
- Community: Greedy modularity communities (NetworkX, ~10-20% cross-edges)

Also supports IID vs Non-IID (label-skew) data distributions.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from torch_geometric.data import Data

from ..utils.common import set_seed


@dataclass
class PartitionResult:
    """Graph partition result with precomputed statistics.

    Attributes:
        partition_map: Tensor [N] mapping node -> client_id
        num_clients: Number of clients
        method: Partition method name
        client_sizes: Number of nodes per client
        cross_edge_ratio: Fraction of edges crossing client boundaries
        cross_edge_matrix: Tensor [K x K] counting edges between each client pair
    """
    partition_map: torch.Tensor
    num_clients: int
    method: str
    client_sizes: List[int]
    cross_edge_ratio: float
    cross_edge_matrix: torch.Tensor


def partition_graph(
    data: Data,
    num_clients: int,
    method: str = 'metis',
    seed: int = 42,
    distribution: str = 'iid',
    label_skew: float = 0.5,
) -> PartitionResult:
    """
    Main partition entry point.

    Args:
        data: Full graph Data object
        num_clients: Number of federated clients (K)
        method: 'metis', 'random', or 'community'
        seed: Random seed
        distribution: 'iid' or 'label_skew'
        label_skew: Skew parameter for non-IID (only used when distribution='label_skew')

    Returns:
        PartitionResult with partition map and statistics
    """
    set_seed(seed)

    if distribution == 'label_skew':
        partition_map = _create_label_skew_partition(data, num_clients, label_skew, seed)
    elif method == 'metis':
        partition_map = _partition_metis(data, num_clients, seed)
    elif method == 'random':
        partition_map = _partition_random(data, num_clients, seed)
    elif method == 'community':
        partition_map = _partition_community(data, num_clients, seed)
    else:
        raise ValueError(f"Unknown partition method: {method}")

    # Compute statistics
    client_sizes = []
    for k in range(num_clients):
        client_sizes.append(int((partition_map == k).sum().item()))

    cross_edge_ratio, cross_edge_matrix = _compute_cross_edge_stats(
        data.edge_index, partition_map, num_clients
    )

    effective_method = f"{method}+{distribution}" if distribution == 'label_skew' else method

    return PartitionResult(
        partition_map=partition_map,
        num_clients=num_clients,
        method=effective_method,
        client_sizes=client_sizes,
        cross_edge_ratio=cross_edge_ratio,
        cross_edge_matrix=cross_edge_matrix,
    )


def _partition_metis(data: Data, num_clients: int, seed: int) -> torch.Tensor:
    """Metis partitioning with NetworkX community detection fallback."""
    try:
        import pymetis
        return _partition_metis_native(data, num_clients, seed)
    except ImportError:
        pass

    # Fallback: NetworkX greedy modularity communities
    return _partition_community(data, num_clients, seed)


def _partition_metis_native(data: Data, num_clients: int, seed: int) -> torch.Tensor:
    """Native Metis partitioning via pymetis."""
    import pymetis

    num_nodes = data.num_nodes
    row, col = data.edge_index.cpu()

    # Build adjacency list for pymetis
    adjacency = [[] for _ in range(num_nodes)]
    for r, c in zip(row.tolist(), col.tolist()):
        if r != c:  # skip self-loops
            adjacency[r].append(c)

    _, membership = pymetis.part_graph(num_clients, adjacency=adjacency)
    partition_map = torch.tensor(membership, dtype=torch.long, device=data.x.device)
    return partition_map


def _partition_community(data: Data, num_clients: int, seed: int) -> torch.Tensor:
    """Community-based partitioning using NetworkX greedy modularity."""
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    num_nodes = data.num_nodes
    row, col = data.edge_index.cpu()

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(row.tolist(), col.tolist()))
    G.add_edges_from(edges)

    # Get communities
    communities = list(greedy_modularity_communities(G))

    # If more communities than clients, merge smallest ones
    # If fewer, split largest ones
    partition_map = torch.full((num_nodes,), -1, dtype=torch.long)

    if len(communities) >= num_clients:
        # Sort communities by size (descending) and assign top-K
        communities = sorted(communities, key=len, reverse=True)
        for k in range(num_clients):
            if k < len(communities):
                for node in communities[k]:
                    partition_map[node] = k

        # Assign remaining nodes to smallest client
        unassigned = (partition_map == -1).nonzero(as_tuple=True)[0]
        if len(unassigned) > 0:
            sizes = torch.bincount(partition_map[partition_map >= 0], minlength=num_clients)
            for node in unassigned:
                smallest = sizes.argmin().item()
                partition_map[node] = smallest
                sizes[smallest] += 1
    else:
        # Fewer communities than clients: assign communities first, then split
        for k, comm in enumerate(communities):
            for node in comm:
                partition_map[node] = k

        # Redistribute from largest community to fill remaining clients
        for k in range(len(communities), num_clients):
            sizes = torch.bincount(partition_map[partition_map >= 0], minlength=num_clients)
            largest = sizes.argmax().item()
            largest_nodes = (partition_map == largest).nonzero(as_tuple=True)[0]
            # Move half to new client
            n_move = max(1, len(largest_nodes) // 2)
            np.random.seed(seed + k)
            move_indices = np.random.choice(largest_nodes.cpu().numpy(), n_move, replace=False)
            partition_map[torch.tensor(move_indices)] = k

    partition_map = partition_map.to(data.x.device)
    return partition_map


def _partition_random(data: Data, num_clients: int, seed: int) -> torch.Tensor:
    """Uniform random node assignment."""
    set_seed(seed)
    num_nodes = data.num_nodes
    partition_map = torch.randint(0, num_clients, (num_nodes,), device=data.x.device)
    return partition_map


def _create_label_skew_partition(
    data: Data,
    num_clients: int,
    skew: float = 0.5,
    seed: int = 42
) -> torch.Tensor:
    """
    Non-IID label-skewed partitioning.

    Each client gets a disproportionate share of 1-2 label classes.
    skew controls the degree: 1.0 = fully separated, 0.0 = IID.
    """
    set_seed(seed)
    num_nodes = data.num_nodes
    labels = data.y.cpu().numpy()
    num_classes = int(data.y.max().item()) + 1

    partition_map = torch.full((num_nodes,), -1, dtype=torch.long)

    # Assign primary class(es) to each client
    rng = np.random.RandomState(seed)
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}

    # Determine how many nodes from each class go to each client
    for c in range(num_classes):
        indices = class_indices[c].copy()
        rng.shuffle(indices)

        # Primary client for this class
        primary_client = c % num_clients

        # Split: skew fraction goes to primary, rest distributed
        n_primary = int(len(indices) * skew)
        n_rest = len(indices) - n_primary

        # Assign primary portion
        partition_map[torch.tensor(indices[:n_primary])] = primary_client

        # Distribute rest uniformly
        if n_rest > 0:
            rest_indices = indices[n_primary:]
            rest_clients = rng.randint(0, num_clients, size=n_rest)
            partition_map[torch.tensor(rest_indices)] = torch.tensor(rest_clients, dtype=torch.long)

    # Handle any unassigned nodes
    unassigned = (partition_map == -1).nonzero(as_tuple=True)[0]
    if len(unassigned) > 0:
        partition_map[unassigned] = torch.randint(0, num_clients, (len(unassigned),))

    partition_map = partition_map.to(data.x.device)
    return partition_map


def _compute_cross_edge_stats(
    edge_index: torch.Tensor,
    partition_map: torch.Tensor,
    num_clients: int
) -> tuple:
    """Compute cross-client edge ratio and pairwise edge count matrix."""
    row, col = edge_index.cpu()
    pm = partition_map.cpu()

    row_client = pm[row]
    col_client = pm[col]

    # Cross-edge ratio
    cross_mask = row_client != col_client
    total_edges = edge_index.size(1)
    cross_edge_ratio = float(cross_mask.sum().item()) / total_edges if total_edges > 0 else 0.0

    # Pairwise cross-edge matrix [K x K]
    cross_matrix = torch.zeros(num_clients, num_clients, dtype=torch.long)
    cross_rows = row_client[cross_mask]
    cross_cols = col_client[cross_mask]
    for r, c in zip(cross_rows.tolist(), cross_cols.tolist()):
        cross_matrix[r, c] += 1

    return cross_edge_ratio, cross_matrix
