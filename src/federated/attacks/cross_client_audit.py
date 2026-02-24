"""
Cross-client leakage analysis for federated graph unlearning.

Quantifies how unlearning information propagates across client boundaries
through FedAvg parameter sharing, and correlates leakage with cross-client
edge connectivity.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from torch_geometric.data import Data

from ..server import SystemSnapshot
from ..subgraph import SubgraphResult
from ..data_partition import PartitionResult
from .hub_ripple_federated import multilevel_hub_ripple, MultiLevelAttackResult


def cross_client_leakage_matrix(
    snapshot_before: SystemSnapshot,
    snapshot_after: SystemSnapshot,
    full_data: Data,
    client_subgraphs: Dict[int, SubgraphResult],
    partition_result: PartitionResult,
    target_idx: int,
    target_client_id: int,
    hub_indices: List[int],
    control_indices: List[int],
    model_template: torch.nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    """
    Compute the cross-client leakage matrix and correlation analysis.

    Returns a comprehensive analysis of how leakage varies across clients
    and correlates with graph connectivity.

    Args:
        snapshot_before/after: System snapshots for comparison
        full_data: Complete graph
        client_subgraphs: All client subgraphs
        partition_result: Partition statistics (includes cross-edge matrix)
        target_idx: Target node global ID
        target_client_id: Client owning target
        hub_indices: Hub node global IDs
        control_indices: Control node global IDs
        model_template: Model architecture
        device: Computation device

    Returns:
        Dict with:
          - leakage_matrix: Dict[client_id -> L2 AUC]
          - cross_edge_counts: Dict[client_id -> num cross-edges to target client]
          - correlation: Pearson r between cross-edges and L2 AUC
          - threshold: Minimum cross-edges for detectable leakage (AUC > 0.55)
    """
    # Run multi-level attack
    attack_result = multilevel_hub_ripple(
        snapshot_before, snapshot_after,
        full_data, client_subgraphs,
        target_idx, target_client_id,
        hub_indices, control_indices,
        model_template, device,
    )

    # Extract cross-edge counts from partition result
    cross_matrix = partition_result.cross_edge_matrix
    cross_edge_counts = {}
    for cid in client_subgraphs:
        if cid == target_client_id:
            continue
        # Edges between target client and this client (bidirectional)
        count = (cross_matrix[target_client_id, cid].item() +
                 cross_matrix[cid, target_client_id].item())
        cross_edge_counts[cid] = count

    # Compute correlation between cross-edges and L2 AUC
    leakage = attack_result.cross_client_l2_aucs
    common_clients = sorted(set(leakage.keys()) & set(cross_edge_counts.keys()))

    correlation = 0.0
    threshold = -1
    if len(common_clients) >= 3:
        edges = [cross_edge_counts[c] for c in common_clients]
        aucs = [leakage[c] for c in common_clients]

        if np.std(edges) > 0 and np.std(aucs) > 0:
            correlation = float(np.corrcoef(edges, aucs)[0, 1])

        # Find minimum cross-edges for detectable leakage
        for c in common_clients:
            if leakage[c] > 0.55 and (threshold < 0 or cross_edge_counts[c] < threshold):
                threshold = cross_edge_counts[c]

    return {
        'leakage_matrix': leakage,
        'cross_edge_counts': cross_edge_counts,
        'correlation': correlation,
        'threshold': threshold,
        'attack_result': attack_result,
    }


def analyze_leakage_by_hops(
    full_data: Data,
    target_idx: int,
    partition_result: PartitionResult,
    client_l2_aucs: Dict[int, float],
) -> Dict[str, object]:
    """
    Analyze how leakage varies with graph distance from the target.

    Groups clients by their minimum hop distance to the target node
    and computes average leakage per distance group.

    Args:
        full_data: Complete graph
        target_idx: Target node
        partition_result: Partition map
        client_l2_aucs: Per-client L2 AUC scores

    Returns:
        Dict with distance-grouped statistics
    """
    from ...utils.graph_utils import get_k_hop_neighbors_fast

    # Get hop distances from target
    hop_neighbors = get_k_hop_neighbors_fast(
        full_data.edge_index, target_idx, k=5, num_nodes=full_data.num_nodes
    )

    # Map nodes to their hop distance
    node_distances = {target_idx: 0}
    for hop, nodes in enumerate(hop_neighbors):
        for n in nodes.tolist():
            node_distances[n] = hop + 1

    # For each client, compute minimum hop distance
    pm = partition_result.partition_map
    client_min_hops = {}
    for cid in client_l2_aucs:
        client_nodes = (pm == cid).nonzero(as_tuple=True)[0].tolist()
        min_hop = min(
            (node_distances.get(n, 999) for n in client_nodes),
            default=999,
        )
        client_min_hops[cid] = min_hop

    # Group by hop distance
    hop_groups = {}
    for cid, hop in client_min_hops.items():
        if hop not in hop_groups:
            hop_groups[hop] = []
        hop_groups[hop].append(client_l2_aucs[cid])

    hop_stats = {}
    for hop in sorted(hop_groups.keys()):
        aucs = hop_groups[hop]
        hop_stats[hop] = {
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'n_clients': len(aucs),
        }

    return {
        'hop_stats': hop_stats,
        'client_min_hops': client_min_hops,
    }
