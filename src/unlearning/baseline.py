"""
Baseline unlearning method: complete retraining.

This is the ground truth unlearning method that completely removes
the target node by retraining from scratch without its edges.
"""

import torch
import copy
from torch_geometric.data import Data

from ..models.gcn import GCN2Layer, GCN3Layer
from ..models.training import train_node_classifier
from ..utils.common import set_seed, DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY, DEFAULT_EPOCHS


def unlearn_baseline(
    model: torch.nn.Module,
    data: Data,
    target_idx: int,
    num_features: int,
    num_classes: int,
    device: torch.device,
    seed: int = 42,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    use_3_layer: bool = False
) -> tuple[torch.nn.Module, Data]:
    """
    Baseline unlearning: Retrain model from scratch without target edges.

    This is the gold standard for unlearning - completely removes all
    information about the target node.

    Args:
        model: Original trained model
        data: Original graph data
        target_idx: Node to unlearn
        num_features: Number of input features
        num_classes: Number of output classes
        device: Device (CPU/GPU)
        seed: Random seed for reproducibility
        epochs: Training epochs (default: 200)
        learning_rate: Learning rate (default: 0.01)
        weight_decay: Weight decay (default: 5e-4)
        use_3_layer: Use 3-layer GCN instead of 2-layer

    Returns:
        Tuple of (unlearned_model, unlearned_data)

    Example:
        >>> model_unlearned, data_unlearned = unlearn_baseline(
        ...     model, data, target_idx, 1433, 7, device
        ... )
    """
    # Remove all edges connected to target
    row, col = data.edge_index
    mask = (row != target_idx) & (col != target_idx)
    edge_index_unlearned = data.edge_index[:, mask]

    # Create unlearned data
    data_unlearned = copy.copy(data)
    data_unlearned.edge_index = edge_index_unlearned

    # Create and train new model from scratch
    if use_3_layer:
        model_new = GCN3Layer(num_features, num_classes).to(device)
    else:
        model_new = GCN2Layer(num_features, num_classes).to(device)

    set_seed(seed)
    model_new.reset_parameters()

    optimizer = torch.optim.Adam(
        model_new.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    train_node_classifier(model_new, data_unlearned, optimizer, epochs=epochs)

    return model_new, data_unlearned
