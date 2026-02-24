"""
GNNDelete unlearning method.

Reference: "GNNDelete: A General Strategy for Unlearning in Graph Neural Networks"
Cheng et al., ICLR 2023 (arXiv:2302.13406)

Core algorithm:
  1. Freeze base GNN weights
  2. Add trainable deletion operators (linear projections) per layer
  3. Optimize two losses:
     - L_DEC (Deleted Edge Consistency): deleted edge endpoints should look like
       random non-edge pairs
     - L_NI (Neighborhood Influence): neighborhood representations should be preserved
  4. Combined loss: L = lambda * L_DEC + (1-lambda) * L_NI

Implementation note: For FedAvg compatibility in the federated setting, we cannot
add extra deletion operator parameters. Instead, we implement two modes:
  - operator mode (default): uses separate deletion operators, returns a GNNDeleteModel
  - compatible mode (use_compatible=True): fine-tunes the GCN with DEC+NI loss,
    returns a standard GCN model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.data import Data

from ..models.gcn import GCN2Layer, GCN3Layer
from ..utils.common import DEFAULT_LEARNING_RATE, DEFAULT_UNLEARN_STEPS


class GNNDeleteModel(nn.Module):
    """GCN with layer-wise deletion operators.

    Wraps a frozen GCN and adds trainable linear deletion operators
    after each convolutional layer. Only the deletion operators are
    trained during unlearning.
    """

    def __init__(self, base_model: nn.Module, affected_mask: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('affected_mask', affected_mask)

        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Deletion operators (linear projections per layer)
        hidden_dim = base_model.conv1.out_channels
        out_dim = base_model.conv2.out_channels

        self.del_op1 = nn.Linear(hidden_dim, hidden_dim)
        self.del_op2 = nn.Linear(out_dim, out_dim)

        # Initialize as identity mapping
        nn.init.eye_(self.del_op1.weight)
        nn.init.zeros_(self.del_op1.bias)
        nn.init.eye_(self.del_op2.weight)
        nn.init.zeros_(self.del_op2.bias)

    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        # Layer 1: conv1 + ReLU + dropout
        h = self.base_model.conv1(x, edge_index, edge_weight)
        h = h.relu()
        h = F.dropout(h, p=self.base_model.dropout, training=self.training)

        # Apply deletion operator 1 to affected nodes only
        h_del = h.clone()
        h_del[self.affected_mask] = self.del_op1(h[self.affected_mask])

        if return_hidden:
            return h_del

        # Layer 2: conv2
        out = self.base_model.conv2(h_del, edge_index, edge_weight)

        # Apply deletion operator 2 to affected nodes
        out_del = out.clone()
        out_del[self.affected_mask] = self.del_op2(out[self.affected_mask])

        return out_del


def unlearn_gnndelete(
    model: torch.nn.Module,
    data: Data,
    target_idx: int,
    num_features: int,
    num_classes: int,
    device: torch.device,
    steps: int = DEFAULT_UNLEARN_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    use_3_layer: bool = False,
    lambda_dec: float = 0.5,
    use_compatible: bool = True,
) -> torch.nn.Module:
    """
    GNNDelete unlearning (Cheng et al., ICLR 2023).

    Optimizes Deleted Edge Consistency (DEC) and Neighborhood Influence (NI).

    Args:
        model: Original trained model
        data: Original graph data
        target_idx: Node to unlearn
        num_features: Number of input features
        num_classes: Number of output classes
        device: Device (CPU/GPU)
        steps: Number of unlearning steps (default: 20)
        learning_rate: Learning rate (default: 0.01)
        use_3_layer: Use 3-layer GCN instead of 2-layer
        lambda_dec: Balance between DEC and NI (default: 0.5)
        use_compatible: If True, return a standard GCN model (for FedAvg);
                        if False, return GNNDeleteModel with deletion operators

    Returns:
        Unlearned model
    """
    # Get target's neighbors (affected nodes)
    row, col = data.edge_index
    neighbors = col[row == target_idx].unique()

    # 2-hop neighborhood for affected mask
    hop2 = set(neighbors.tolist())
    for n in neighbors.tolist():
        n_neighbors = col[row == n].tolist()
        hop2.update(n_neighbors)
    hop2.add(target_idx)
    affected_indices = sorted(hop2)
    affected_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    affected_mask[affected_indices] = True

    # Get original embeddings as reference for NI loss
    model.eval()
    with torch.no_grad():
        orig_emb = model(data.x, data.edge_index)

    # Remove target's edges from graph
    edge_mask = (row != target_idx) & (col != target_idx)
    clean_edge_index = data.edge_index[:, edge_mask]

    # Deleted edges: all edges incident to target
    deleted_edges_mask = ~edge_mask
    deleted_u = data.edge_index[0, deleted_edges_mask]
    deleted_v = data.edge_index[1, deleted_edges_mask]

    if use_compatible:
        # Compatible mode: fine-tune GCN with DEC+NI loss
        return _unlearn_compatible(
            model, data, target_idx, num_features, num_classes, device,
            steps, learning_rate, use_3_layer, lambda_dec,
            orig_emb, clean_edge_index, neighbors, affected_mask,
            deleted_u, deleted_v,
        )
    else:
        # Operator mode: use GNNDeleteModel
        return _unlearn_operator(
            model, data, target_idx, device,
            steps, learning_rate, lambda_dec,
            orig_emb, clean_edge_index, neighbors, affected_mask,
            deleted_u, deleted_v,
        )


def _unlearn_compatible(
    model, data, target_idx, num_features, num_classes, device,
    steps, learning_rate, use_3_layer, lambda_dec,
    orig_emb, clean_edge_index, neighbors, affected_mask,
    deleted_u, deleted_v,
):
    """Fine-tune GCN model with DEC+NI loss (standard architecture output)."""
    if use_3_layer:
        model_new = GCN3Layer(num_features, num_classes).to(device)
    else:
        model_new = GCN2Layer(num_features, num_classes).to(device)

    model_new.load_state_dict(copy.deepcopy(model.state_dict()))

    optimizer = torch.optim.Adam(model_new.parameters(), lr=learning_rate)

    num_deleted = len(deleted_u)

    for step in range(steps):
        optimizer.zero_grad()
        model_new.train()

        # Forward on clean graph (target edges removed)
        out = model_new(data.x, clean_edge_index)

        # === L_DEC: Deleted Edge Consistency ===
        # Deleted edge endpoints should look like random non-edge pairs
        if num_deleted > 0:
            # Concatenated representations of deleted edge endpoints
            del_rep = torch.cat([out[deleted_u], out[deleted_v]], dim=1)

            # Random non-edge pair representations (detached as reference)
            rand_u = torch.randint(0, data.num_nodes, (num_deleted,), device=device)
            rand_v = torch.randint(0, data.num_nodes, (num_deleted,), device=device)
            with torch.no_grad():
                ref_rep = torch.cat([out[rand_u], out[rand_v]], dim=1)

            loss_dec = F.mse_loss(del_rep, ref_rep)
        else:
            loss_dec = torch.tensor(0.0, device=device)

        # === L_NI: Neighborhood Influence ===
        # Neighborhood representations should be preserved
        non_target = affected_mask.clone()
        non_target[target_idx] = False
        if non_target.any():
            loss_ni = F.mse_loss(out[non_target], orig_emb[non_target].detach())
        else:
            loss_ni = torch.tensor(0.0, device=device)

        # Combined loss
        loss = lambda_dec * loss_dec + (1 - lambda_dec) * loss_ni
        loss.backward()
        optimizer.step()

    return model_new


def _unlearn_operator(
    model, data, target_idx, device,
    steps, learning_rate, lambda_dec,
    orig_emb, clean_edge_index, neighbors, affected_mask,
    deleted_u, deleted_v,
):
    """Use GNNDeleteModel with separate deletion operators."""
    base_model = copy.deepcopy(model).to(device)
    del_model = GNNDeleteModel(base_model, affected_mask).to(device)

    # Only optimize deletion operator parameters
    optimizer = torch.optim.Adam(
        list(del_model.del_op1.parameters()) + list(del_model.del_op2.parameters()),
        lr=learning_rate,
    )

    num_deleted = len(deleted_u)

    for step in range(steps):
        optimizer.zero_grad()
        del_model.train()

        out = del_model(data.x, clean_edge_index)

        # L_DEC
        if num_deleted > 0:
            del_rep = torch.cat([out[deleted_u], out[deleted_v]], dim=1)
            rand_u = torch.randint(0, data.num_nodes, (num_deleted,), device=device)
            rand_v = torch.randint(0, data.num_nodes, (num_deleted,), device=device)
            with torch.no_grad():
                ref_rep = torch.cat([out[rand_u], out[rand_v]], dim=1)
            loss_dec = F.mse_loss(del_rep, ref_rep)
        else:
            loss_dec = torch.tensor(0.0, device=device)

        # L_NI
        non_target = affected_mask.clone()
        non_target[target_idx] = False
        if non_target.any():
            loss_ni = F.mse_loss(out[non_target], orig_emb[non_target].detach())
        else:
            loss_ni = torch.tensor(0.0, device=device)

        loss = lambda_dec * loss_dec + (1 - lambda_dec) * loss_ni
        loss.backward()
        optimizer.step()

    return del_model
