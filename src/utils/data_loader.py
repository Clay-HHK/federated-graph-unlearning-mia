"""
Dataset loading utilities.

This module provides unified dataset loading for all experiments,
including homophily computation and data preprocessing.
"""

import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.data import Data
from typing import Tuple

from .common import get_dataset_path


def load_dataset(name: str, device: torch.device) -> Tuple[Data, float]:
    """
    Load a graph dataset and compute its homophily ratio.

    Args:
        name: Dataset name ('Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel')
        device: Device to load data onto

    Returns:
        Tuple of (data, homophily_ratio)

    Example:
        >>> device = torch.device('cuda')
        >>> data, homophily = load_dataset('Cora', device)
        >>> print(f"Cora homophily: {homophily:.3f}")
    """
    # Load dataset
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=get_dataset_path(name), name=name)
    elif name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root=get_dataset_path(name), name=name.lower())
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data = dataset[0].to(device)

    # Fix train_mask dimension for Chameleon/Squirrel
    if name in ['Chameleon', 'Squirrel']:
        if data.train_mask.dim() > 1:
            data.train_mask = data.train_mask[:, 0]
        if hasattr(data, 'val_mask') and data.val_mask.dim() > 1:
            data.val_mask = data.val_mask[:, 0]
        if hasattr(data, 'test_mask') and data.test_mask.dim() > 1:
            data.test_mask = data.test_mask[:, 0]

    # Compute homophily
    homophily = compute_homophily(data)

    return data, homophily


def compute_homophily(data: Data) -> float:
    """
    Compute edge homophily ratio.

    Homophily = (# edges connecting same-label nodes) / (# total edges)

    Args:
        data: PyTorch Geometric data object

    Returns:
        Homophily ratio [0, 1]

    Example:
        >>> homophily = compute_homophily(data)
        >>> print(f"Homophily: {homophily:.3f}")
    """
    row, col = data.edge_index
    same_label = (data.y[row] == data.y[col]).sum().item()
    total_edges = data.edge_index.size(1)
    homophily = same_label / total_edges if total_edges > 0 else 0.0
    return homophily


def get_dataset_stats(name: str, device: torch.device) -> dict:
    """
    Get comprehensive dataset statistics.

    Args:
        name: Dataset name
        device: Device to load data onto

    Returns:
        Dictionary with dataset statistics

    Example:
        >>> stats = get_dataset_stats('Cora', device)
        >>> print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
    """
    data, homophily = load_dataset(name, device)

    stats = {
        'name': name,
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.size(1),
        'num_features': data.num_features,
        'num_classes': data.y.max().item() + 1,
        'homophily': homophily,
        'avg_degree': data.edge_index.size(1) / data.num_nodes,
    }

    return stats


def print_dataset_info(name: str, device: torch.device):
    """
    Print formatted dataset information.

    Args:
        name: Dataset name
        device: Device to load data onto

    Example:
        >>> print_dataset_info('Cora', device)
        Dataset: Cora
        Nodes: 2708, Edges: 10556
        Features: 1433, Classes: 7
        Homophily: 0.810
        Avg Degree: 3.90
    """
    stats = get_dataset_stats(name, device)

    print(f"\nDataset: {stats['name']}")
    print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
    print(f"Features: {stats['num_features']}, Classes: {stats['num_classes']}")
    print(f"Homophily: {stats['homophily']:.3f}")
    print(f"Avg Degree: {stats['avg_degree']:.2f}\n")
