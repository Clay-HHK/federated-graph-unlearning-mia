"""
Training utilities for GNN models.

This module provides standard training procedures used across
all Hub-Ripple MIA experiments.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def train_node_classifier(
    model: torch.nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    epochs: int = 200,
    verbose: bool = False
) -> torch.nn.Module:
    """
    Train a node classification model.

    Args:
        model: GNN model to train
        data: PyTorch Geometric data object with x, edge_index, y, train_mask
        optimizer: Optimizer (typically Adam)
        epochs: Number of training epochs (default: 200)
        verbose: Print training loss (default: False)

    Returns:
        Trained model

    Example:
        >>> model = GCN(in_channels=1433, out_channels=7)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        >>> model = train_node_classifier(model, data, optimizer, epochs=200)
    """
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x, data.edge_index)

        # Compute loss on training nodes only
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model


def get_embeddings(model: torch.nn.Module, data) -> torch.Tensor:
    """
    Extract node embeddings from trained model.

    Automatically detects and uses edge_weight if present in data
    (e.g., from GraphEditor unlearning).

    Args:
        model: Trained GNN model
        data: PyTorch Geometric data object (may contain edge_weight)

    Returns:
        Node embeddings [num_nodes, out_channels]

    Example:
        >>> embeddings = get_embeddings(model, data)
        >>> hub_embedding = embeddings[hub_idx]
    """
    model.eval()
    with torch.no_grad():
        # Support GraphEditor's learned edge_weight
        edge_weight = getattr(data, 'edge_weight', None)
        if edge_weight is not None:
            embeddings = model(data.x, data.edge_index, edge_weight=edge_weight)
        else:
            embeddings = model(data.x, data.edge_index)
    return embeddings


def get_embeddings_at_layer(model: torch.nn.Module, data, layer: str = 'logits') -> torch.Tensor:
    """
    Extract node embeddings at a specified representation layer.

    Args:
        model: Trained GNN model (must support return_hidden parameter)
        data: PyTorch Geometric data object (may contain edge_weight)
        layer: One of 'hidden' (penultimate), 'logits', 'softmax'

    Returns:
        Node embeddings at the specified layer

    Example:
        >>> hidden_emb = get_embeddings_at_layer(model, data, layer='hidden')
        >>> logit_emb = get_embeddings_at_layer(model, data, layer='logits')
        >>> prob_emb = get_embeddings_at_layer(model, data, layer='softmax')
    """
    model.eval()
    with torch.no_grad():
        edge_weight = getattr(data, 'edge_weight', None)
        kwargs = {}
        if edge_weight is not None:
            kwargs['edge_weight'] = edge_weight

        if layer == 'hidden':
            embeddings = model(data.x, data.edge_index, return_hidden=True, **kwargs)
        elif layer == 'logits':
            embeddings = model(data.x, data.edge_index, **kwargs)
        elif layer == 'softmax':
            logits = model(data.x, data.edge_index, **kwargs)
            embeddings = F.softmax(logits, dim=1)
        else:
            raise ValueError(f"Unknown layer: {layer}. Use 'hidden', 'logits', or 'softmax'.")

    return embeddings


def train_with_edge_weights(
    model: torch.nn.Module,
    data,
    edge_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int = 200,
    verbose: bool = False
) -> torch.nn.Module:
    """
    Train a model with learnable edge weights.

    Used by GraphEditor unlearning method.

    Args:
        model: GNN model to train
        data: PyTorch Geometric data object
        edge_weights: Learnable edge weight tensor [num_edges]
        optimizer: Optimizer for both model and edge weights
        epochs: Number of training epochs (default: 200)
        verbose: Print training loss (default: False)

    Returns:
        Trained model
    """
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass with edge weights
        out = model(data.x, data.edge_index, edge_weight=edge_weights)

        # Compute loss
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Clamp edge weights to [0, 1]
        with torch.no_grad():
            edge_weights.clamp_(0.0, 1.0)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, data, mask: Optional[torch.Tensor] = None) -> float:
    """
    Evaluate model accuracy.

    Automatically detects and uses edge_weight if present in data
    (e.g., from GraphEditor unlearning).

    Args:
        model: Trained GNN model
        data: PyTorch Geometric data object (may contain edge_weight)
        mask: Node mask for evaluation (default: data.test_mask)

    Returns:
        Accuracy score
    """
    model.eval()

    if mask is None:
        mask = data.test_mask

    # Support GraphEditor's learned edge_weight
    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is not None:
        out = model(data.x, data.edge_index, edge_weight=edge_weight)
    else:
        out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    accuracy = correct.sum().item() / mask.sum().item()

    return accuracy


# Alias for backward compatibility with federated learning code
train_model = train_node_classifier
