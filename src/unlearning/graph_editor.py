"""
GraphEditor unlearning method.

Reference: "Graph Unlearning" (Chen et al., arXiv 2022)

GraphEditor learns edge weights to "soft-delete" target nodes by:
1. Freezing model parameters
2. Learning weights on target's edges
3. Minimizing target's prediction confidence

This is a lightweight unlearning method that doesn't require retraining.
"""

import torch
import torch.nn.functional as F
import copy
from torch_geometric.data import Data
from typing import Tuple

from ..utils.common import DEFAULT_UNLEARN_STEPS


def unlearn_graph_editor(
    model: torch.nn.Module,
    data: Data,
    target_idx: int,
    device: torch.device,
    steps: int = DEFAULT_UNLEARN_STEPS,
    learning_rate: float = 0.1,
    return_weights: bool = False
) -> Tuple[torch.nn.Module, Data, torch.Tensor]:
    """
    GraphEditor unlearning via learnable edge weights.

    Learns weights on target's edges to minimize target's influence
    without modifying model parameters.

    Args:
        model: Original trained model
        data: Original graph data
        target_idx: Node to unlearn
        device: Device (CPU/GPU)
        steps: Number of optimization steps (default: 20)
        learning_rate: Learning rate for edge weights (default: 0.1)
        return_weights: Whether to return learned edge weights

    Returns:
        Tuple of (model, data_with_weights, edge_weights) if return_weights=True
        Otherwise (model, data, None)

    Example:
        >>> model_unlearned, data, weights = unlearn_graph_editor(
        ...     model, data, target_idx, device, steps=20
        ... )
    """
    # Find target's edges
    row, col = data.edge_index
    target_edge_mask = (row == target_idx) | (col == target_idx)
    edge_mask_indices = torch.where(target_edge_mask)[0]

    if len(edge_mask_indices) == 0:
        # No edges to learn, return original model
        return model, data, None

    # Initialize learnable weights for target edges
    # Start with 1.0 (full weight), will learn to reduce
    weights = torch.ones(len(edge_mask_indices), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([weights], lr=learning_rate)

    # Freeze model parameters
    model_copy = copy.deepcopy(model)
    for p in model_copy.parameters():
        p.requires_grad = False

    # Learn edge weights
    for step in range(steps):
        optimizer.zero_grad()
        model_copy.train()

        # Create full weight vector
        full_weights = torch.ones(data.edge_index.size(1), device=device)
        full_weights[edge_mask_indices] = weights

        # Forward pass with weighted edges
        out = model_copy(data.x, data.edge_index, edge_weight=full_weights)

        # Loss: maximize uncertainty on target (minimize confidence)
        # This encourages the model to "forget" the target
        loss = -F.cross_entropy(
            out[target_idx].unsqueeze(0),
            data.y[target_idx].unsqueeze(0)
        )

        loss.backward()
        optimizer.step()

        # Clamp weights to [0, 1]
        with torch.no_grad():
            weights.clamp_(0.0, 1.0)

    # Store learned weights
    learned_weights = weights.detach().clone()

    # Create data object with learned edge weights
    # For Hub-Ripple attack, we'll use these weights during embedding extraction
    if return_weights:
        # Create full weight tensor
        full_edge_weights = torch.ones(data.edge_index.size(1), device=device)
        full_edge_weights[edge_mask_indices] = learned_weights

        # Store in data object
        data_weighted = copy.copy(data)
        data_weighted.edge_weight = full_edge_weights

        return model_copy, data_weighted, learned_weights
    else:
        return model_copy, data, None


def unlearn_graph_editor_with_neighbors(
    model: torch.nn.Module,
    data: Data,
    target_idx: int,
    device: torch.device,
    steps: int = DEFAULT_UNLEARN_STEPS,
    learning_rate: float = 0.1,
    neighbor_retention_weight: float = 0.5
) -> Tuple[torch.nn.Module, Data]:
    """
    Enhanced GraphEditor with neighbor retention.

    Balances between forgetting target and retaining neighbor information.

    Args:
        model: Original trained model
        data: Original graph data
        target_idx: Node to unlearn
        device: Device (CPU/GPU)
        steps: Number of optimization steps (default: 20)
        learning_rate: Learning rate for edge weights (default: 0.1)
        neighbor_retention_weight: Weight for neighbor retention loss (default: 0.5)

    Returns:
        Tuple of (model, data_with_weights)

    Example:
        >>> model_unlearned, data = unlearn_graph_editor_with_neighbors(
        ...     model, data, target_idx, device
        ... )
    """
    # Find target's edges and neighbors
    row, col = data.edge_index
    target_edge_mask = (row == target_idx) | (col == target_idx)
    edge_mask_indices = torch.where(target_edge_mask)[0]

    # Get target's neighbors
    neighbors = col[row == target_idx]

    if len(edge_mask_indices) == 0:
        return model, data

    # Initialize learnable weights
    weights = torch.ones(len(edge_mask_indices), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([weights], lr=learning_rate)

    # Freeze model parameters
    model_copy = copy.deepcopy(model)
    for p in model_copy.parameters():
        p.requires_grad = False

    # Learn edge weights with dual objective
    for step in range(steps):
        optimizer.zero_grad()
        model_copy.train()

        # Create full weight vector
        full_weights = torch.ones(data.edge_index.size(1), device=device)
        full_weights[edge_mask_indices] = weights

        # Forward pass
        out = model_copy(data.x, data.edge_index, edge_weight=full_weights)

        # Loss 1: Forget target
        loss_forget = -F.cross_entropy(
            out[target_idx].unsqueeze(0),
            data.y[target_idx].unsqueeze(0)
        )

        # Loss 2: Retain neighbors
        if len(neighbors) > 0:
            loss_retain = F.cross_entropy(out[neighbors], data.y[neighbors])
        else:
            loss_retain = torch.tensor(0.0, device=device)

        # Combined loss
        loss = loss_forget + neighbor_retention_weight * loss_retain

        loss.backward()
        optimizer.step()

        # Clamp weights
        with torch.no_grad():
            weights.clamp_(0.0, 1.0)

    # Create data with learned weights
    learned_weights = weights.detach().clone()
    full_edge_weights = torch.ones(data.edge_index.size(1), device=device)
    full_edge_weights[edge_mask_indices] = learned_weights

    data_weighted = copy.copy(data)
    data_weighted.edge_weight = full_edge_weights

    return model_copy, data_weighted


# Alias for backward compatibility
unlearn_grapheditor = unlearn_graph_editor
