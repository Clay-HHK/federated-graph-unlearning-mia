"""
GIF (Graph Influence Function) unlearning method.

Reference: "GIF: A General Graph Unlearning Strategy via Influence Function"
Wu et al., WWW 2023 (arXiv:2304.02835)

Core algorithm:
  1. Compute influence of target node on model parameters using influence functions
  2. Include neighbor influence: k-hop neighbors' prediction changes due to removal
  3. Approximate Hessian inverse via Neumann series (iterative HVP)
  4. Update parameters: theta_new = theta + H^{-1} * grad_delta_L

The influence function formula:
  theta_hat - theta_0 = H^{-1} * nabla Δℒ

where Δℒ includes:
  - Loss on target node (to be removed)
  - Loss change on k-hop neighbors (structural dependency)

Hessian inverse approximation (Neumann series):
  H^{-1}v ≈ sum_{j=0}^{J} (I - lambda*H)^j * v
  Iterative: r_j = v + r_{j-1} - lambda * H * r_{j-1}
"""

import torch
import torch.nn.functional as F
import copy
from torch_geometric.data import Data

from ..models.gcn import GCN2Layer, GCN3Layer
from ..utils.common import DEFAULT_LEARNING_RATE


def _compute_hvp(model, data, v, train_mask=None, damping=0.01):
    """
    Compute Hessian-vector product H*v using autograd.

    Uses the double backward trick:
      grad = autograd.grad(loss, params, create_graph=True)
      hvp = autograd.grad(grad · v, params)

    Args:
        model: The model
        data: Graph data
        v: Vector to multiply with Hessian (list of tensors matching params)
        train_mask: Training mask for loss computation
        damping: Damping factor for numerical stability

    Returns:
        List of tensors: H*v + damping*v
    """
    model.zero_grad()
    out = model(data.x, data.edge_index)

    if train_mask is not None and train_mask.any():
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    else:
        loss = F.cross_entropy(out, data.y)

    # First-order gradients with computation graph retained
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Compute dot product: grad · v
    gv = sum((g * vi).sum() for g, vi in zip(grads, v))

    # Second-order: Hessian-vector product
    hvp = torch.autograd.grad(gv, params)

    # Add damping for stability: (H + damping*I) * v
    return [h.detach() + damping * vi for h, vi in zip(hvp, v)]


def _neumann_ihvp(model, data, v, train_mask=None, iterations=5, damping=0.01, scale=1.0):
    """
    Approximate H^{-1}*v using Neumann series.

    H^{-1}v ≈ sum_{j=0}^{J} (I - scale*H)^j * v

    Iterative formulation:
      r_0 = v
      r_j = v + r_{j-1} - scale * H * r_{j-1}

    Args:
        model: The model
        data: Graph data
        v: Vector to compute H^{-1}*v for
        train_mask: Training mask
        iterations: Number of Neumann iterations (default: 5)
        damping: Damping factor
        scale: Scaling factor (typically 1/n for n training samples)

    Returns:
        Approximation of H^{-1}*v as list of tensors
    """
    r = [vi.clone() for vi in v]  # r_0 = v

    for _ in range(iterations):
        # Compute H * r_{j-1}
        try:
            hvp = _compute_hvp(model, data, r, train_mask, damping=0.0)
        except RuntimeError:
            # If second-order gradients fail, fall back to identity approximation
            return [vi * scale for vi in v]

        # r_j = v + r_{j-1} - scale * H * r_{j-1}
        r = [vi + ri - scale * hi for vi, ri, hi in zip(v, r, hvp)]

    return r


def unlearn_gif(
    model: torch.nn.Module,
    data: Data,
    target_idx: int,
    num_features: int,
    num_classes: int,
    device: torch.device,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    use_3_layer: bool = False,
    hessian_iterations: int = 5,
    damping: float = 0.01,
    include_neighbors: bool = True,
) -> torch.nn.Module:
    """
    GIF unlearning using influence function approximation (Wu et al., WWW 2023).

    Computes the influence of the target node (and its neighbors) on model
    parameters, then removes that influence via a Hessian-scaled parameter update.

    The influence function formula:
      theta_new = theta + H^{-1} * nabla_Delta_L

    where Delta_L includes:
      - Gradient on target node loss (direct influence)
      - Gradient changes on k-hop neighbors (structural dependency)

    Args:
        model: Original trained model
        data: Original graph data
        target_idx: Node to unlearn
        num_features: Number of input features
        num_classes: Number of output classes
        device: Device (CPU/GPU)
        learning_rate: Step size for influence removal (default: 0.01)
        use_3_layer: Use 3-layer GCN instead of 2-layer
        hessian_iterations: Number of Neumann series iterations (default: 5)
        damping: Hessian damping factor (default: 0.01)
        include_neighbors: Include neighbor influence terms (default: True)

    Returns:
        Unlearned model
    """
    # Create new model and copy weights
    if use_3_layer:
        model_new = GCN3Layer(num_features, num_classes).to(device)
    else:
        model_new = GCN2Layer(num_features, num_classes).to(device)

    model_new.load_state_dict(copy.deepcopy(model.state_dict()))

    # Enable gradient computation
    for p in model_new.parameters():
        p.requires_grad = True

    # === Step 1: Compute gradient on target node (direct influence) ===
    model_new.train()
    out = model_new(data.x, data.edge_index)
    loss_target = F.cross_entropy(
        out[target_idx].unsqueeze(0),
        data.y[target_idx].unsqueeze(0),
    )
    grad_target = torch.autograd.grad(
        loss_target, model_new.parameters(), retain_graph=True,
    )
    grad_target = [g.detach().clone() for g in grad_target]

    # === Step 2: Compute neighbor influence (structural dependency) ===
    grad_neighbor_delta = None
    if include_neighbors:
        row, col = data.edge_index
        neighbors = col[row == target_idx].unique()

        if len(neighbors) > 0:
            # Loss on neighbors with original graph (before edge removal)
            loss_nei_before = F.cross_entropy(out[neighbors], data.y[neighbors])
            grad_nei_before = torch.autograd.grad(
                loss_nei_before, model_new.parameters(), retain_graph=True,
            )
            grad_nei_before = [g.detach().clone() for g in grad_nei_before]

            # Loss on neighbors after removing target's edges
            edge_mask = (row != target_idx) & (col != target_idx)
            clean_edge_index = data.edge_index[:, edge_mask]

            out_clean = model_new(data.x, clean_edge_index)
            loss_nei_after = F.cross_entropy(
                out_clean[neighbors], data.y[neighbors],
            )
            grad_nei_after = torch.autograd.grad(
                loss_nei_after, model_new.parameters(),
            )
            grad_nei_after = [g.detach().clone() for g in grad_nei_after]

            # Delta gradient from neighbors
            grad_neighbor_delta = [
                gb - ga for gb, ga in zip(grad_nei_before, grad_nei_after)
            ]

    # === Step 3: Combine gradients (target + neighbor delta) ===
    if grad_neighbor_delta is not None:
        total_grad = [
            gt + gn for gt, gn in zip(grad_target, grad_neighbor_delta)
        ]
    else:
        total_grad = grad_target

    # === Step 4: Approximate H^{-1} * total_grad ===
    train_mask = getattr(data, 'train_mask', None)
    n_train = train_mask.sum().item() if train_mask is not None else data.num_nodes
    scale = 1.0 / max(n_train, 1)

    try:
        # Neumann series approximation of H^{-1} * grad
        model_ref = copy.deepcopy(model_new)
        for p in model_ref.parameters():
            p.requires_grad = True

        ihvp = _neumann_ihvp(
            model_ref, data, total_grad,
            train_mask=train_mask,
            iterations=hessian_iterations,
            damping=damping,
            scale=scale,
        )
    except (RuntimeError, ValueError):
        # Fallback: approximate H^{-1} ≈ learning_rate * I
        ihvp = [g * learning_rate for g in total_grad]

    # === Step 5: Apply parameter update ===
    # theta_new = theta + H^{-1} * grad (add, since we want to REMOVE influence)
    # The sign convention: loss gradient points toward the target's influence,
    # so we subtract to remove it
    with torch.no_grad():
        for p, h in zip(model_new.parameters(), ihvp):
            p.add_(h, alpha=-1.0)

    return model_new
