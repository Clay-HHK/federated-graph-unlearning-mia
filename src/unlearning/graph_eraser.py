"""
GraphEraser unlearning method.

Reference: "Graph Unlearning" (Chen et al., CCS 2022, arXiv:2103.14991)

GraphEraser is an enhanced SISA method that:
1. Uses feature-based balanced partitioning (similar to BEKM)
2. Trains independent models per shard
3. Provides certified deletion guarantees
4. Aggregates predictions via ensemble

Key difference from standard SISA: GraphEraser focuses on certified
guarantees and optimal partition strategies for graphs.
"""

import torch
import numpy as np
import copy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.data import Data
from typing import List, Tuple

from ..models.gcn import GCN3Layer, GCN2Layer
from ..models.training import train_node_classifier
from ..utils.common import (
    set_seed,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_EPOCHS,
    DEFAULT_NUM_SHARDS,
    DEFAULT_BALANCE_RATIO
)
from ..utils.graph_utils import filter_edges_by_nodes_vectorized, remap_edge_indices


def partition_graph_eraser(
    data: Data,
    num_shards: int = DEFAULT_NUM_SHARDS,
    balance_ratio: float = DEFAULT_BALANCE_RATIO,
    seed: int = 42
) -> torch.Tensor:
    """
    GraphEraser partitioning strategy.

    Uses balanced K-Means with additional heuristics for better
    partition quality.

    Args:
        data: PyTorch Geometric data object
        num_shards: Number of shards (default: 5)
        balance_ratio: Max shard size ratio (default: 1.1)
        seed: Random seed

    Returns:
        Shard assignment tensor [num_nodes]
    """
    set_seed(seed)

    features = data.x.cpu().numpy()
    num_nodes = data.num_nodes
    max_size = int(np.ceil((num_nodes / num_shards) * balance_ratio))

    # K-Means with multiple initializations for stability
    best_kmeans = None
    best_inertia = float('inf')

    for trial in range(5):
        kmeans = KMeans(
            n_clusters=num_shards,
            random_state=seed + trial,
            n_init=20,
            max_iter=300
        )
        kmeans.fit(features)

        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_kmeans = kmeans

    # Balanced assignment with priority queue
    dists = euclidean_distances(features, best_kmeans.cluster_centers_)

    shard_map = -np.ones(num_nodes, dtype=int)
    counts = np.zeros(num_shards, dtype=int)

    # Sort by minimum distance (closer nodes get priority)
    node_order = np.argsort(np.min(dists, axis=1))

    for idx in node_order:
        # Assign to nearest available cluster
        for k in np.argsort(dists[idx]):
            if counts[k] < max_size:
                shard_map[idx] = k
                counts[k] += 1
                break

        # If all clusters full, assign to least full
        if shard_map[idx] == -1:
            k = np.argmin(counts)
            shard_map[idx] = k
            counts[k] += 1

    shard_map = torch.tensor(shard_map, dtype=torch.long, device=data.x.device)
    return shard_map


def train_shard_model(
    shard_data: Data,
    num_features: int,
    num_classes: int,
    device: torch.device,
    shard_id: int,
    seed: int = 42,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    use_3_layer: bool = True
) -> torch.nn.Module:
    """
    Train a model for a single shard.

    Args:
        shard_data: Shard-specific graph data
        num_features: Number of input features
        num_classes: Number of output classes
        device: Device (CPU/GPU)
        shard_id: Shard ID (for seed offset)
        seed: Base random seed
        epochs: Training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_3_layer: Use 3-layer GCN

    Returns:
        Trained shard model
    """
    # Create model
    if use_3_layer:
        model = GCN3Layer(num_features, num_classes).to(device)
    else:
        model = GCN2Layer(num_features, num_classes).to(device)

    # Reset with shard-specific seed for independence
    set_seed(seed + shard_id)
    model.reset_parameters()

    # Train
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    train_node_classifier(model, shard_data, optimizer, epochs=epochs)

    return model


def build_shard_subgraph(
    data: Data,
    shard_map: torch.Tensor,
    shard_id: int,
    exclude_node: int = None
) -> Data:
    """
    Extract subgraph for a specific shard using vectorized operations.

    Args:
        data: Original graph data
        shard_map: Shard assignment
        shard_id: Target shard ID
        exclude_node: Node to exclude (for unlearning)

    Returns:
        Shard subgraph data
    """
    device = data.x.device

    # Get shard nodes
    shard_nodes = torch.where(shard_map == shard_id)[0]

    # Exclude target if specified (vectorized)
    if exclude_node is not None:
        shard_nodes = shard_nodes[shard_nodes != exclude_node]

    if len(shard_nodes) == 0:
        raise ValueError(f"Shard {shard_id} is empty")

    # Use vectorized edge filtering - O(E) instead of O(E×N)
    edge_mask = filter_edges_by_nodes_vectorized(data.edge_index, shard_nodes, device)

    # Remap to local indices using vectorized operation
    local_edge_index = remap_edge_indices(data.edge_index, edge_mask, shard_nodes, device)

    # Handle empty edge case: create minimal self-loop
    if local_edge_index.size(1) == 0:
        local_edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)

    # Create shard data
    shard_data = Data(
        x=data.x[shard_nodes],
        y=data.y[shard_nodes],
        edge_index=local_edge_index,
        train_mask=data.train_mask[shard_nodes]
    )
    shard_data.num_nodes = len(shard_nodes)

    return shard_data


def unlearn_graph_eraser(
    model: torch.nn.Module,
    data: Data,
    target_idx: int,
    num_features: int,
    num_classes: int,
    device: torch.device,
    num_shards: int = DEFAULT_NUM_SHARDS,
    balance_ratio: float = DEFAULT_BALANCE_RATIO,
    seed: int = 42,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    use_3_layer: bool = True
) -> Tuple[torch.nn.Module, Data]:
    """
    GraphEraser certified unlearning.

    Implements the GraphEraser algorithm with optimized partitioning
    and shard retraining.

    Args:
        model: Original trained model
        data: Original graph data
        target_idx: Node to unlearn
        num_features: Number of input features
        num_classes: Number of output classes
        device: Device (CPU/GPU)
        num_shards: Number of shards (default: 5)
        balance_ratio: Balance constraint (default: 1.1)
        seed: Random seed
        epochs: Training epochs (default: 200)
        learning_rate: Learning rate (default: 0.01)
        weight_decay: Weight decay (default: 5e-4)
        use_3_layer: Use 3-layer GCN

    Returns:
        Tuple of (unlearned_model, original_data)

    Example:
        >>> model_unlearned, data = unlearn_graph_eraser(
        ...     model, data, target_idx, 1433, 7, device
        ... )
    """
    # Partition graph using GraphEraser strategy
    shard_map = partition_graph_eraser(data, num_shards, balance_ratio, seed)

    # Find target's shard
    target_shard = shard_map[target_idx].item()

    # Build shard subgraph (excluding target)
    shard_data = build_shard_subgraph(data, shard_map, target_shard, exclude_node=target_idx)

    # Retrain affected shard
    model_new = train_shard_model(
        shard_data,
        num_features,
        num_classes,
        device,
        target_shard,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_3_layer=use_3_layer
    )

    return model_new, data


def unlearn_graph_eraser_ensemble(
    models: List[torch.nn.Module],
    data: Data,
    target_idx: int,
    num_features: int,
    num_classes: int,
    device: torch.device,
    num_shards: int = DEFAULT_NUM_SHARDS,
    balance_ratio: float = DEFAULT_BALANCE_RATIO,
    seed: int = 42,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    use_3_layer: bool = True
) -> Tuple[List[torch.nn.Module], Data]:
    """
    GraphEraser with full ensemble (all K shard models).

    Returns all shard models for ensemble prediction.

    Args:
        models: List of original shard models
        data: Original graph data
        target_idx: Node to unlearn
        num_features: Number of input features
        num_classes: Number of output classes
        device: Device (CPU/GPU)
        num_shards: Number of shards
        balance_ratio: Balance constraint
        seed: Random seed
        epochs: Training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_3_layer: Use 3-layer GCN

    Returns:
        Tuple of (list_of_shard_models, original_data)

    Example:
        >>> models_unlearned, data = unlearn_graph_eraser_ensemble(
        ...     [None]*5, data, target_idx, 1433, 7, device
        ... )
    """
    # Partition graph
    shard_map = partition_graph_eraser(data, num_shards, balance_ratio, seed)

    # Find target's shard
    target_shard = shard_map[target_idx].item()

    # Initialize ensemble models
    ensemble_models = []

    for shard_id in range(num_shards):
        if shard_id == target_shard:
            # Retrain affected shard (excluding target)
            shard_data = build_shard_subgraph(data, shard_map, shard_id, exclude_node=target_idx)
            model = train_shard_model(
                shard_data, num_features, num_classes, device, shard_id,
                seed, epochs, learning_rate, weight_decay, use_3_layer
            )
        else:
            # Keep original shard model (if provided, otherwise train new)
            if models and shard_id < len(models) and models[shard_id] is not None:
                model = models[shard_id]
            else:
                shard_data = build_shard_subgraph(data, shard_map, shard_id)
                model = train_shard_model(
                    shard_data, num_features, num_classes, device, shard_id,
                    seed, epochs, learning_rate, weight_decay, use_3_layer
                )

        ensemble_models.append(model)

    return ensemble_models, data
