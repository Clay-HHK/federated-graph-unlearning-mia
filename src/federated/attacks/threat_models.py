"""
Threat model definitions for federated graph unlearning privacy audit.

Four threat models with decreasing adversary capability:
- TM1 (White-Box): Full information access (academic benchmark)
- TM2 (Local Auditor): Access to own client + global model (practical)
- TM3 (Server Auditor): Access to global model + API (server-side)
- TM4 (Black-Box): Prediction API only (weakest)
"""

import torch
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional
from torch_geometric.data import Data

from ..server import SystemSnapshot
from ..subgraph import SubgraphResult, filter_nodes_in_client
from .hub_ripple_federated import (
    _load_model, _compute_drifts, _compute_conf_drifts, _safe_auc
)
from ...models.training import get_embeddings


@dataclass
class ThreatModelResult:
    """Result from a threat model evaluation."""
    threat_model: str
    l2_auc: float
    conf_auc: float
    available_levels: List[str]
    detail: Dict[str, float]


class ThreatModel:
    """Base threat model class."""
    name: str = "base"

    def can_access_global_model(self) -> bool:
        return False

    def can_access_local_models(self) -> bool:
        return False

    def can_access_client_data(self) -> bool:
        return False

    def can_access_embeddings(self) -> bool:
        return False

    def can_access_predictions_only(self) -> bool:
        return True


class TM1_WhiteBox(ThreatModel):
    """White-box: full information access.

    Academic benchmark scenario. The adversary has access to:
    - Global model weights (before and after)
    - All client local models
    - All client data and graph structure
    - Raw embeddings at all levels

    Executes the full multi-level audit.
    """
    name = "TM1_WhiteBox"

    def can_access_global_model(self) -> bool:
        return True

    def can_access_local_models(self) -> bool:
        return True

    def can_access_client_data(self) -> bool:
        return True

    def can_access_embeddings(self) -> bool:
        return True


class TM2_LocalAuditor(ThreatModel):
    """Local auditor: a federation participant.

    Most practical threat model. The adversary is a client who:
    - Can observe the global model (training participant)
    - Has access to its own subgraph data
    - Cannot access other clients' data or local models
    - Can detect remote unlearning through global model changes

    Available levels: Global (via global model), own-client Local.
    """
    name = "TM2_LocalAuditor"

    def __init__(self, auditor_client_id: int):
        self.auditor_client_id = auditor_client_id

    def can_access_global_model(self) -> bool:
        return True

    def can_access_local_models(self) -> bool:
        return False  # Only own local model

    def can_access_client_data(self) -> bool:
        return False  # Only own data

    def can_access_embeddings(self) -> bool:
        return True


class TM3_ServerAuditor(ThreatModel):
    """Server auditor: the aggregation server.

    The server can:
    - Access global model weights and aggregation metadata
    - Query the model for embeddings on any data it holds
    - Cannot access clients' raw data

    Available levels: Global only (server doesn't hold subgraph data).
    """
    name = "TM3_ServerAuditor"

    def can_access_global_model(self) -> bool:
        return True

    def can_access_embeddings(self) -> bool:
        return True


class TM4_BlackBox(ThreatModel):
    """Black-box: prediction API only.

    Weakest adversary. Can only:
    - Query the model for predictions (confidence scores)
    - Cannot access model weights or embeddings

    Expected: Completely ineffective (Conf AUC ~ 0.5).
    Used to prove the confidence illusion.
    """
    name = "TM4_BlackBox"

    def can_access_predictions_only(self) -> bool:
        return True


def evaluate_threat_model(
    tm: ThreatModel,
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
) -> ThreatModelResult:
    """
    Evaluate a threat model's attack capability.

    Executes only the attack components accessible to the given threat model.

    Args:
        tm: Threat model instance
        snapshot_before/after: System snapshots
        full_data: Complete graph
        client_subgraphs: Client subgraphs
        target_idx: Target node
        target_client_id: Target's client
        hub_indices: Hub nodes
        control_indices: Control nodes
        model_template: Model architecture
        device: Device

    Returns:
        ThreatModelResult with accessible metrics
    """
    detail = {}
    available_levels = []
    best_l2_auc = 0.5
    best_conf_auc = 0.5

    full_data = full_data.to(device)

    # ===== Global level (TM1, TM2, TM3) =====
    if tm.can_access_global_model() and tm.can_access_embeddings():
        model_before = _load_model(model_template, snapshot_before.global_state, device)
        model_after = _load_model(model_template, snapshot_after.global_state, device)

        emb_before = get_embeddings(model_before, full_data)
        emb_after = get_embeddings(model_after, full_data)

        hub_l2 = _compute_drifts(emb_before, emb_after, hub_indices, 'l2')
        ctrl_l2 = _compute_drifts(emb_before, emb_after, control_indices, 'l2')
        global_l2 = _safe_auc(hub_l2, ctrl_l2)

        hub_conf = _compute_conf_drifts(emb_before, emb_after, hub_indices)
        ctrl_conf = _compute_conf_drifts(emb_before, emb_after, control_indices)
        global_conf = _safe_auc(hub_conf, ctrl_conf)

        detail['global_l2_auc'] = global_l2
        detail['global_conf_auc'] = global_conf
        available_levels.append('global')
        best_l2_auc = max(best_l2_auc, global_l2)
        best_conf_auc = max(best_conf_auc, global_conf)

    # ===== Local level (TM1 only, or TM2 if auditor == target client) =====
    can_local = False
    if isinstance(tm, TM1_WhiteBox):
        can_local = True
    elif isinstance(tm, TM2_LocalAuditor) and tm.auditor_client_id == target_client_id:
        can_local = True

    if can_local and target_client_id in client_subgraphs:
        local_before = snapshot_before.local_states.get(target_client_id)
        local_after = snapshot_after.local_states.get(target_client_id)

        if local_before is not None and local_after is not None:
            subgraph = client_subgraphs[target_client_id]
            model_before = _load_model(model_template, local_before, device)
            model_after = _load_model(model_template, local_after, device)

            local_data = subgraph.data.to(device)
            emb_before = get_embeddings(model_before, local_data)
            emb_after = get_embeddings(model_after, local_data)

            _, local_hubs = filter_nodes_in_client(hub_indices, subgraph)
            _, local_ctrls = filter_nodes_in_client(control_indices, subgraph)

            if len(local_hubs) >= 2 and len(local_ctrls) >= 2:
                hub_l2 = _compute_drifts(emb_before, emb_after, local_hubs, 'l2')
                ctrl_l2 = _compute_drifts(emb_before, emb_after, local_ctrls, 'l2')
                local_l2 = _safe_auc(hub_l2, ctrl_l2)

                hub_conf = _compute_conf_drifts(emb_before, emb_after, local_hubs)
                ctrl_conf = _compute_conf_drifts(emb_before, emb_after, local_ctrls)
                local_conf = _safe_auc(hub_conf, ctrl_conf)

                detail['local_l2_auc'] = local_l2
                detail['local_conf_auc'] = local_conf
                available_levels.append('local')
                best_l2_auc = max(best_l2_auc, local_l2)
                best_conf_auc = max(best_conf_auc, local_conf)

    # ===== Cross-client level (TM2 from its own perspective) =====
    if isinstance(tm, TM2_LocalAuditor) and tm.auditor_client_id != target_client_id:
        auditor_subgraph = client_subgraphs.get(tm.auditor_client_id)
        if auditor_subgraph is not None:
            model_before = _load_model(model_template, snapshot_before.global_state, device)
            model_after = _load_model(model_template, snapshot_after.global_state, device)

            local_data = auditor_subgraph.data.to(device)
            emb_before = get_embeddings(model_before, local_data)
            emb_after = get_embeddings(model_after, local_data)

            _, local_hubs = filter_nodes_in_client(hub_indices, auditor_subgraph)
            _, local_ctrls = filter_nodes_in_client(control_indices, auditor_subgraph)

            if len(local_hubs) >= 2 and len(local_ctrls) >= 2:
                hub_l2 = _compute_drifts(emb_before, emb_after, local_hubs, 'l2')
                ctrl_l2 = _compute_drifts(emb_before, emb_after, local_ctrls, 'l2')
                cross_l2 = _safe_auc(hub_l2, ctrl_l2)

                detail['cross_l2_auc'] = cross_l2
                available_levels.append('cross_client')
                best_l2_auc = max(best_l2_auc, cross_l2)

    # ===== Black-box: confidence only =====
    if isinstance(tm, TM4_BlackBox):
        model_before = _load_model(model_template, snapshot_before.global_state, device)
        model_after = _load_model(model_template, snapshot_after.global_state, device)

        emb_before = get_embeddings(model_before, full_data)
        emb_after = get_embeddings(model_after, full_data)

        hub_conf = _compute_conf_drifts(emb_before, emb_after, hub_indices)
        ctrl_conf = _compute_conf_drifts(emb_before, emb_after, control_indices)
        bb_conf = _safe_auc(hub_conf, ctrl_conf)

        detail['blackbox_conf_auc'] = bb_conf
        available_levels.append('blackbox')
        best_conf_auc = bb_conf
        best_l2_auc = 0.5  # Not accessible

    return ThreatModelResult(
        threat_model=tm.name,
        l2_auc=best_l2_auc,
        conf_auc=best_conf_auc,
        available_levels=available_levels,
        detail=detail,
    )
