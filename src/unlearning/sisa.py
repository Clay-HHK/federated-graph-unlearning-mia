"""
SISA (Sharded, Isolated, Sliced, Aggregated) unlearning methods.

Reference: "Machine Unlearning" (Bourtoule et al., S&P 2021)

SISA partitions the graph into K shards and trains independent models.
When unlearning, only the affected shard is retrained.

We implement three partitioning strategies:
- BEKM (Balanced K-Means): Feature-based balanced clustering
- BLPA (Balanced Label Propagation): Community-based partitioning
- Random: Uniform random assignment
"""

import torch
import numpy as np
import copy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.data import Data
from typing import Tuple

from ..models.gcn import GCN3Layer, GCN2Layer
from ..models.training import train_node_classifier
from ..utils.common import set_seed, DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY, DEFAULT_EPOCHS, DEFAULT_NUM_SHARDS, DEFAULT_BALANCE_RATIO
from ..utils.graph_utils import filter_edges_by_nodes_vectorized, remap_edge_indices


def partition_bekm(
    data: Data,
    num_shards: int = DEFAULT_NUM_SHARDS,
    balance_ratio: float = DEFAULT_BALANCE_RATIO,
    seed: int = 42
) -> torch.Tensor:
    """
    Balanced K-Means (BEKM) graph partitioning.

    Uses K-Means on node features with balance constraints to ensure
    shards have similar sizes.

    Args:
        data: PyTorch Geometric data object
        num_shards: Number of shards (default: 5)
        balance_ratio: Max shard size = (num_nodes / K) * balance_ratio (default: 1.1)
        seed: Random seed

    Returns:
        Shard assignment tensor [num_nodes]

    Example:
        >>> shard_map = partition_bekm(data, num_shards=5)
        >>> target_shard = shard_map[target_idx].item()
    """
    set_seed(seed)

    features = data.x.cpu().numpy()
    num_nodes = data.num_nodes
    max_size = int(np.ceil((num_nodes / num_shards) * balance_ratio))

    # K-Means clustering
    kmeans = KMeans(n_clusters=num_shards, random_state=seed, n_init=10)
    kmeans.fit(features)

    # Compute distances to cluster centers
    dists = euclidean_distances(features, kmeans.cluster_centers_)

    # Balanced assignment
    shard_map = -np.ones(num_nodes, dtype=int)
    counts = np.zeros(num_shards, dtype=int)

    # Sort nodes by minimum distance to any cluster
    for idx in np.argsort(np.min(dists, axis=1)):
        # Assign to nearest cluster with space
        for k in np.argsort(dists[idx]):
            if counts[k] < max_size:
                shard_map[idx] = k
                counts[k] += 1
                break

    shard_map = torch.tensor(shard_map, dtype=torch.long, device=data.x.device)
    return shard_map


def partition_blpa(
    data: Data,
    num_shards: int = DEFAULT_NUM_SHARDS,
    balance_ratio: float = DEFAULT_BALANCE_RATIO,
    max_iters: int = 10,
    seed: int = 42,
    convergence_threshold: float = 0.01
) -> torch.Tensor:
    """
    Balanced Label Propagation Algorithm (BLPA) graph partitioning.

    Uses community detection with balance constraints to preserve
    graph structure. Optimized with:
    - Random traversal order to avoid order-dependent bias
    - Pre-computed shard sizes to avoid repeated counting
    - Convergence detection for early stopping

    Args:
        data: PyTorch Geometric data object
        num_shards: Number of shards (default: 5)
        balance_ratio: Max shard size = (num_nodes / K) * balance_ratio (default: 1.1)
        max_iters: Maximum label propagation iterations (default: 10)
        seed: Random seed
        convergence_threshold: Stop if fewer than this fraction of nodes change (default: 0.01)

    Returns:
        Shard assignment tensor [num_nodes]

    Example:
        >>> shard_map = partition_blpa(data, num_shards=5)
    """
    set_seed(seed)

    num_nodes = data.num_nodes
    device = data.x.device
    max_size = int(np.ceil((num_nodes / num_shards) * balance_ratio))

    # Initialize with random labels
    labels = torch.randint(0, num_shards, (num_nodes,), device=device)

    # Pre-build adjacency list for O(1) neighbor lookup
    row, col = data.edge_index
    adj_list = [[] for _ in range(num_nodes)]
    for r, c in zip(row.tolist(), col.tolist()):
        adj_list[r].append(c)

    # Pre-compute initial shard sizes
    shard_sizes = torch.bincount(labels, minlength=num_shards).tolist()

    # Label propagation with optimizations
    for iteration in range(max_iters):
        new_labels = labels.clone()
        changes = 0

        # Random traversal order to avoid order-dependent bias
        node_order = torch.randperm(num_nodes, device=device).tolist()

        for node in node_order:
            neighbors = adj_list[node]
            if len(neighbors) == 0:
                continue

            # Count neighbor labels efficiently
            neighbor_labels = labels[neighbors]
            label_counts = torch.bincount(neighbor_labels, minlength=num_shards)

            # Get current label
            current_label = labels[node].item()

            # Find best label (most frequent among neighbors, respecting balance)
            for most_freq_label in torch.argsort(label_counts, descending=True):
                most_freq_label = most_freq_label.item()

                # Skip if same as current
                if most_freq_label == current_label:
                    break

                # Check balance constraint using pre-computed sizes
                if shard_sizes[most_freq_label] < max_size:
                    # Update labels and sizes
                    new_labels[node] = most_freq_label
                    shard_sizes[current_label] -= 1
                    shard_sizes[most_freq_label] += 1
                    changes += 1
                    break

        labels = new_labels

        # Convergence check: stop if few nodes changed
        change_ratio = changes / num_nodes
        if change_ratio < convergence_threshold:
            break

    return labels


def partition_random(
    data: Data,
    num_shards: int = DEFAULT_NUM_SHARDS,
    seed: int = 42
) -> torch.Tensor:
    """
    Random graph partitioning.

    Uniformly randomly assigns nodes to shards.

    Args:
        data: PyTorch Geometric data object
        num_shards: Number of shards (default: 5)
        seed: Random seed

    Returns:
        Shard assignment tensor [num_nodes]

    Example:
        >>> shard_map = partition_random(data, num_shards=5)
    """
    set_seed(seed)
    num_nodes = data.num_nodes
    shard_map = torch.randint(0, num_shards, (num_nodes,), device=data.x.device)
    return shard_map


def build_shard_data(
    data: Data,
    shard_map: torch.Tensor,
    shard_id: int,
    exclude_node: int = None
) -> Data:
    """
    Build data for a specific shard using vectorized operations.

    Args:
        data: Original graph data
        shard_map: Shard assignment [num_nodes]
        shard_id: Shard ID to extract
        exclude_node: Node to exclude (for unlearning)

    Returns:
        Shard-specific data object

    Example:
        >>> shard_data = build_shard_data(data, shard_map, shard_id=2, exclude_node=100)
    """
    device = data.x.device

    # Get shard nodes
    shard_nodes = torch.where(shard_map == shard_id)[0]

    # Exclude target node if specified
    if exclude_node is not None:
        # Vectorized exclusion
        shard_nodes = shard_nodes[shard_nodes != exclude_node]

    if len(shard_nodes) == 0:
        raise ValueError(f"Shard {shard_id} is empty after excluding node {exclude_node}")

    # Use vectorized edge filtering - O(E) instead of O(E×N)
    edge_mask = filter_edges_by_nodes_vectorized(data.edge_index, shard_nodes, device)

    # Remap to local indices using vectorized operation
    local_edge_index = remap_edge_indices(data.edge_index, edge_mask, shard_nodes, device)

    # Create shard data
    shard_data = Data(
        x=data.x[shard_nodes],
        y=data.y[shard_nodes],
        edge_index=local_edge_index,
        train_mask=data.train_mask[shard_nodes],
        val_mask=data.val_mask[shard_nodes] if hasattr(data, 'val_mask') and data.val_mask is not None else None,
        test_mask=data.test_mask[shard_nodes] if hasattr(data, 'test_mask') and data.test_mask is not None else None
    )
    shard_data.num_nodes = len(shard_nodes)

    return shard_data


def unlearn_sisa(
    model: torch.nn.Module,
    data: Data,
    target_idx: int,
    num_features: int,
    num_classes: int,
    device: torch.device,
    partition_strategy: str = 'bekm',
    num_shards: int = DEFAULT_NUM_SHARDS,
    balance_ratio: float = DEFAULT_BALANCE_RATIO,
    seed: int = 42,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    use_3_layer: bool = True
) -> Tuple[torch.nn.Module, Data]:
    """
    SISA unlearning with configurable partitioning strategy.

    Args:
        model: Original trained model
        data: Original graph data
        target_idx: Node to unlearn
        num_features: Number of input features
        num_classes: Number of output classes
        device: Device (CPU/GPU)
        partition_strategy: Partitioning method ('bekm', 'blpa', 'random')
        num_shards: Number of shards (default: 5)
        balance_ratio: Balance constraint (default: 1.1)
        seed: Random seed
        epochs: Training epochs (default: 200)
        learning_rate: Learning rate (default: 0.01)
        weight_decay: Weight decay (default: 5e-4)
        use_3_layer: Use 3-layer GCN instead of 2-layer

    Returns:
        Tuple of (unlearned_model, original_data)

    Example:
        >>> model_unlearned, data = unlearn_sisa(
        ...     model, data, target_idx, 1433, 7, device,
        ...     partition_strategy='bekm'
        ... )
    """
    # Partition graph
    if partition_strategy == 'bekm':
        shard_map = partition_bekm(data, num_shards, balance_ratio, seed)
    elif partition_strategy == 'blpa':
        shard_map = partition_blpa(data, num_shards, balance_ratio, seed=seed)
    elif partition_strategy == 'random':
        shard_map = partition_random(data, num_shards, seed)
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")

    # Find target's shard
    target_shard = shard_map[target_idx].item()

    # Build shard data (excluding target)
    shard_data = build_shard_data(data, shard_map, target_shard, exclude_node=target_idx)

    # Retrain shard model
    model_new = GCN3Layer(num_features, num_classes).to(device) if use_3_layer else GCN2Layer(num_features, num_classes).to(device)
    set_seed(seed + target_shard)
    model_new.reset_parameters()

    optimizer = torch.optim.Adam(
        model_new.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    train_node_classifier(model_new, shard_data, optimizer, epochs=epochs)

    # Return model trained on affected shard (for embedding extraction on full graph, we use original data)
    return model_new, data
