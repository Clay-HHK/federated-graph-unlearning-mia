"""
Multi-level Hub-Ripple MIA for federated graph unlearning.

Core contribution: extends the centralized Hub-Ripple attack to three
audit levels in the federated setting:

Level 1 - Global:  Uses global model on full data
Level 2 - Local:   Uses target client's local model on its subgraph
Level 3 - Cross:   Uses global model on other clients' subgraphs

Key insight: FedAvg dilutes confidence signals (Conf AUC ~ 0.5) but
L2 geometric drift persists across all levels.
"""

import torch
import numpy as np
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data

from ..server import SystemSnapshot
from ..subgraph import SubgraphResult, filter_nodes_in_client
from ...attacks.hub_ripple import measure_embedding_drift
from ...attacks.metrics import compute_attack_metrics
from ...models.gcn import GCN2Layer
from ...models.training import get_embeddings


@dataclass
class MultiLevelAttackResult:
    """Multi-level attack results — core contribution data structure.

    Captures privacy leakage at Global, Local, and Cross-Client levels,
    for both L2 geometric and confidence-based metrics.
    """
    # Global level (aggregated model on full graph)
    global_l2_auc: float = 0.5
    global_conf_auc: float = 0.5
    global_gap: float = 0.0

    # Local level (target client's model on its subgraph)
    local_l2_auc: float = 0.5
    local_conf_auc: float = 0.5
    local_gap: float = 0.0

    # Cross-client level (global model on other clients' subgraphs)
    cross_client_l2_aucs: Dict[int, float] = field(default_factory=dict)
    cross_client_conf_aucs: Dict[int, float] = field(default_factory=dict)
    mean_cross_l2_auc: float = 0.5
    max_cross_l2_auc: float = 0.5

    # Metadata
    target_idx: int = -1
    target_client_id: int = -1
    num_hubs: int = 0
    num_controls: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Multi-Level Hub-Ripple Attack Results",
            "=" * 60,
            f"Target Node: {self.target_idx} (Client {self.target_client_id})",
            f"Hubs: {self.num_hubs}, Controls: {self.num_controls}",
            "",
            "Level       | L2 AUC | Conf AUC | Gap",
            "-" * 50,
            f"Global      | {self.global_l2_auc:.4f} | {self.global_conf_auc:.4f}  | {self.global_gap:.4f}",
            f"Local       | {self.local_l2_auc:.4f} | {self.local_conf_auc:.4f}  | {self.local_gap:.4f}",
            f"Cross (avg) | {self.mean_cross_l2_auc:.4f} | -        | -",
            f"Cross (max) | {self.max_cross_l2_auc:.4f} | -        | -",
        ]
        if self.cross_client_l2_aucs:
            lines.append("")
            lines.append("Cross-Client Detail:")
            for cid in sorted(self.cross_client_l2_aucs.keys()):
                l2 = self.cross_client_l2_aucs[cid]
                conf = self.cross_client_conf_aucs.get(cid, float('nan'))
                lines.append(f"  Client {cid}: L2={l2:.4f}, Conf={conf:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


def multilevel_hub_ripple(
    snapshot_before: SystemSnapshot,
    snapshot_after: SystemSnapshot,
    full_data: Data,
    client_subgraphs: Dict[int, SubgraphResult],
    target_idx: int,
    target_client_id: int,
    hub_indices: List[int],
    control_indices: List[int],
    model_template: torch.nn.Module,
    device: torch.device,
    metric: str = 'l2',
) -> MultiLevelAttackResult:
    """
    Execute Hub-Ripple MIA at three audit levels.

    Args:
        snapshot_before: System state before unlearning
        snapshot_after: System state after unlearning
        full_data: Complete graph data (for global-level evaluation)
        client_subgraphs: Dict mapping client_id -> SubgraphResult
        target_idx: Global node ID of the unlearning target
        target_client_id: Client that owns the target node
        hub_indices: Global node IDs of hub nodes (target's neighbors)
        control_indices: Global node IDs of control nodes (non-neighbors)
        model_template: Model architecture template for loading states
        device: Computation device
        metric: Distance metric ('l2', 'l1', 'cosine')

    Returns:
        MultiLevelAttackResult with all three levels
    """
    result = MultiLevelAttackResult(
        target_idx=target_idx,
        target_client_id=target_client_id,
        num_hubs=len(hub_indices),
        num_controls=len(control_indices),
    )

    # ===== Level 1: Global =====
    global_l2, global_conf = _attack_global(
        snapshot_before, snapshot_after,
        full_data, hub_indices, control_indices,
        model_template, device, metric,
    )
    result.global_l2_auc = global_l2
    result.global_conf_auc = global_conf
    result.global_gap = global_l2 - global_conf

    # ===== Level 2: Local (target client) =====
    if target_client_id in client_subgraphs:
        local_l2, local_conf = _attack_local(
            snapshot_before, snapshot_after,
            client_subgraphs[target_client_id],
            target_client_id,
            hub_indices, control_indices,
            model_template, device, metric,
        )
        result.local_l2_auc = local_l2
        result.local_conf_auc = local_conf
        result.local_gap = local_l2 - local_conf

    # ===== Level 3: Cross-Client =====
    cross_l2_aucs = {}
    cross_conf_aucs = {}
    for cid, subgraph in client_subgraphs.items():
        if cid == target_client_id:
            continue

        cross_l2, cross_conf = _attack_cross_client(
            snapshot_before, snapshot_after,
            subgraph, hub_indices, control_indices,
            model_template, device, metric,
        )
        if cross_l2 is not None:
            cross_l2_aucs[cid] = cross_l2
            cross_conf_aucs[cid] = cross_conf

    result.cross_client_l2_aucs = cross_l2_aucs
    result.cross_client_conf_aucs = cross_conf_aucs

    if cross_l2_aucs:
        result.mean_cross_l2_auc = float(np.mean(list(cross_l2_aucs.values())))
        result.max_cross_l2_auc = float(max(cross_l2_aucs.values()))
    else:
        result.mean_cross_l2_auc = 0.5
        result.max_cross_l2_auc = 0.5

    return result


def _load_model(template: torch.nn.Module, state: OrderedDict, device: torch.device):
    """Load a model from state_dict using the template architecture."""
    model = copy.deepcopy(template).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _compute_drifts(
    emb_before: torch.Tensor,
    emb_after: torch.Tensor,
    node_indices: List[int],
    metric: str,
) -> List[float]:
    """Compute embedding drift for a list of node indices."""
    drifts = []
    for idx in node_indices:
        drift = measure_embedding_drift(emb_before[idx], emb_after[idx], metric)
        drifts.append(drift)
    return drifts


def _compute_conf_drifts(
    emb_before: torch.Tensor,
    emb_after: torch.Tensor,
    node_indices: List[int],
) -> List[float]:
    """Compute confidence-based drift (max softmax probability change)."""
    import torch.nn.functional as F
    drifts = []
    prob_before = F.softmax(emb_before, dim=1)
    prob_after = F.softmax(emb_after, dim=1)

    for idx in node_indices:
        conf_before = prob_before[idx].max().item()
        conf_after = prob_after[idx].max().item()
        drifts.append(abs(conf_before - conf_after))
    return drifts


def _safe_auc(hub_drifts: List[float], control_drifts: List[float]) -> float:
    """Compute AUC from hub vs control drifts, returning 0.5 if insufficient data."""
    if len(hub_drifts) < 2 or len(control_drifts) < 2:
        return 0.5

    y_true = [1] * len(hub_drifts) + [0] * len(control_drifts)
    y_scores = hub_drifts + control_drifts

    try:
        return float(roc_auc_score(y_true, y_scores))
    except ValueError:
        return 0.5


def _attack_global(
    snap_before: SystemSnapshot,
    snap_after: SystemSnapshot,
    full_data: Data,
    hub_indices: List[int],
    control_indices: List[int],
    model_template: torch.nn.Module,
    device: torch.device,
    metric: str,
) -> tuple:
    """Level 1: Global model on full data."""
    model_before = _load_model(model_template, snap_before.global_state, device)
    model_after = _load_model(model_template, snap_after.global_state, device)

    full_data = full_data.to(device)
    emb_before = get_embeddings(model_before, full_data)
    emb_after = get_embeddings(model_after, full_data)

    # L2 drift
    hub_l2 = _compute_drifts(emb_before, emb_after, hub_indices, metric)
    ctrl_l2 = _compute_drifts(emb_before, emb_after, control_indices, metric)
    l2_auc = _safe_auc(hub_l2, ctrl_l2)

    # Confidence drift
    hub_conf = _compute_conf_drifts(emb_before, emb_after, hub_indices)
    ctrl_conf = _compute_conf_drifts(emb_before, emb_after, control_indices)
    conf_auc = _safe_auc(hub_conf, ctrl_conf)

    return l2_auc, conf_auc


def _attack_local(
    snap_before: SystemSnapshot,
    snap_after: SystemSnapshot,
    target_subgraph: SubgraphResult,
    target_client_id: int,
    hub_indices: List[int],
    control_indices: List[int],
    model_template: torch.nn.Module,
    device: torch.device,
    metric: str,
) -> tuple:
    """Level 2: Target client's local model on its subgraph."""
    # Load local models
    local_state_before = snap_before.local_states.get(target_client_id)
    local_state_after = snap_after.local_states.get(target_client_id)

    if local_state_before is None or local_state_after is None:
        return 0.5, 0.5

    model_before = _load_model(model_template, local_state_before, device)
    model_after = _load_model(model_template, local_state_after, device)

    local_data = target_subgraph.data.to(device)
    emb_before = get_embeddings(model_before, local_data)
    emb_after = get_embeddings(model_after, local_data)

    # Filter hub/control to nodes in this client (convert to local indices)
    _, local_hubs = filter_nodes_in_client(hub_indices, target_subgraph)
    _, local_ctrls = filter_nodes_in_client(control_indices, target_subgraph)

    if len(local_hubs) < 2 or len(local_ctrls) < 2:
        return 0.5, 0.5

    hub_l2 = _compute_drifts(emb_before, emb_after, local_hubs, metric)
    ctrl_l2 = _compute_drifts(emb_before, emb_after, local_ctrls, metric)
    l2_auc = _safe_auc(hub_l2, ctrl_l2)

    hub_conf = _compute_conf_drifts(emb_before, emb_after, local_hubs)
    ctrl_conf = _compute_conf_drifts(emb_before, emb_after, local_ctrls)
    conf_auc = _safe_auc(hub_conf, ctrl_conf)

    return l2_auc, conf_auc


def _attack_cross_client(
    snap_before: SystemSnapshot,
    snap_after: SystemSnapshot,
    other_subgraph: SubgraphResult,
    hub_indices: List[int],
    control_indices: List[int],
    model_template: torch.nn.Module,
    device: torch.device,
    metric: str,
) -> tuple:
    """
    Level 3: Global model on another client's subgraph.

    Uses the global model (not the other client's local model) because
    cross-client information propagates only through FedAvg aggregation.
    """
    model_before = _load_model(model_template, snap_before.global_state, device)
    model_after = _load_model(model_template, snap_after.global_state, device)

    local_data = other_subgraph.data.to(device)
    emb_before = get_embeddings(model_before, local_data)
    emb_after = get_embeddings(model_after, local_data)

    # Filter hub/control to nodes in this other client
    _, local_hubs = filter_nodes_in_client(hub_indices, other_subgraph)
    _, local_ctrls = filter_nodes_in_client(control_indices, other_subgraph)

    if len(local_hubs) < 2 or len(local_ctrls) < 2:
        return None, None

    hub_l2 = _compute_drifts(emb_before, emb_after, local_hubs, metric)
    ctrl_l2 = _compute_drifts(emb_before, emb_after, local_ctrls, metric)
    l2_auc = _safe_auc(hub_l2, ctrl_l2)

    hub_conf = _compute_conf_drifts(emb_before, emb_after, local_hubs)
    ctrl_conf = _compute_conf_drifts(emb_before, emb_after, local_ctrls)
    conf_auc = _safe_auc(hub_conf, ctrl_conf)

    return l2_auc, conf_auc
