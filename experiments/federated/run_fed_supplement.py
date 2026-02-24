"""
Unified Supplementary Experiment: RQ2 + RQ4 + RQ5

Combines three experiments into a single run to avoid redundant training:
  - RQ2: Cross-client leakage correlation (per-client AUC vs cross-edge count)
  - RQ4: Threat model comparison (TM1-TM4)
  - RQ5: Node-level vs client-level unlearning granularity

For each trial, the training phase is shared across all evaluations:
  1. Train federated model once
  2. For each of the 5 node-level methods: unlearn → collect RQ2/RQ4 data
  3. Perform client-level unlearning → collect RQ5 data
  4. Save 3 separate CSV files

Methods: FedRetrain, FedGNNDelete, FedGIF, FedGraphEraser-BEKM, FedGraphEraser-BLPA

Usage:
    python experiments/federated/run_fed_supplement.py --trials 100
    python experiments/federated/run_fed_supplement.py --quick --trials 10
"""

import sys
import os
import time
import copy
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.common import set_seed, get_device
from src.utils.data_loader import load_dataset
from src.utils.graph_utils import (
    get_neighbors, select_control_node, get_node_degrees,
    get_k_hop_neighbors_fast,
)
from src.models.gcn import GCN2Layer
from src.federated.subgraph import build_client_subgraph, filter_nodes_in_client
from src.federated.data_partition import partition_graph
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.unlearning import (
    fed_retrain, fed_gnndelete, fed_gif, fed_graph_eraser, fed_client_unlearning,
)
from src.federated.attacks.hub_ripple_federated import (
    multilevel_hub_ripple, _load_model, _compute_drifts,
    _compute_conf_drifts, _safe_auc,
)
from src.federated.attacks.threat_models import (
    TM1_WhiteBox, TM2_LocalAuditor, TM3_ServerAuditor, TM4_BlackBox,
    evaluate_threat_model,
)
from src.models.training import get_embeddings


# ========================================================================
# Helpers
# ========================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def select_target_and_hubs(data, partition_result, seed=42, min_hubs=5):
    """Select target node and hub/control pairs."""
    set_seed(seed)
    degrees = get_node_degrees(data.edge_index, data.num_nodes)
    sorted_nodes = torch.argsort(degrees, descending=True)

    target_idx = None
    for candidate in sorted_nodes[:100].tolist():
        neighbors = get_neighbors(data.edge_index, candidate)
        if len(neighbors) >= min_hubs:
            target_idx = candidate
            break

    if target_idx is None:
        target_idx = sorted_nodes[0].item()

    target_client_id = partition_result.partition_map[target_idx].item()
    neighbors = get_neighbors(data.edge_index, target_idx)
    hub_indices = neighbors.tolist()

    control_indices = []
    for hub_idx in hub_indices:
        try:
            ctrl = select_control_node(
                data.edge_index, data.num_nodes,
                hub_idx, target_idx, seed=seed + hub_idx
            )
            control_indices.append(ctrl)
        except ValueError:
            continue

    min_len = min(len(hub_indices), len(control_indices))
    return target_idx, target_client_id, hub_indices[:min_len], control_indices[:min_len]


def clone_server(server):
    """Deep copy a server for independent unlearning."""
    ul_clients = []
    for cid in server.client_ids:
        ul_clients.append(FederatedClient(
            cid, copy.deepcopy(server.clients[cid].subgraph),
            server.clients[cid].model, server.device,
        ))
    return FederatedServer(copy.deepcopy(server.global_model), ul_clients, server.device)


def do_unlearn(ul_server, method, target_idx, target_client_id,
               data, partition_result, num_features, num_classes,
               grid, trial_seed):
    """Dispatch unlearning by method name. Returns FedUnlearnResult."""
    if method == 'FedRetrain':
        return fed_retrain(
            ul_server, target_idx, target_client_id,
            data, partition_result, num_features, num_classes,
            num_rounds=grid['num_rounds'], local_epochs=grid['local_epochs'],
            seed=trial_seed + 1000,
        )
    elif method == 'FedGNNDelete':
        return fed_gnndelete(
            ul_server, target_idx, target_client_id,
            num_features, num_classes,
            reaggregate_rounds=5, local_epochs=grid['local_epochs'],
        )
    elif method == 'FedGIF':
        return fed_gif(
            ul_server, target_idx, target_client_id,
            num_features, num_classes,
            reaggregate_rounds=5, local_epochs=grid['local_epochs'],
        )
    elif method == 'FedGraphEraser-BEKM':
        return fed_graph_eraser(
            ul_server, target_idx, target_client_id,
            num_features, num_classes,
            reaggregate_rounds=5, local_epochs=grid['local_epochs'],
            partition_strategy='bekm',
            epochs=grid['epochs'], seed=trial_seed,
        )
    elif method == 'FedGraphEraser-BLPA':
        return fed_graph_eraser(
            ul_server, target_idx, target_client_id,
            num_features, num_classes,
            reaggregate_rounds=5, local_epochs=grid['local_epochs'],
            partition_strategy='blpa',
            epochs=grid['epochs'], seed=trial_seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# ========================================================================
# RQ2: Per-client leakage
# ========================================================================

def collect_rq2(snap_before, snap_after, full_data, client_subgraphs,
                target_idx, target_client_id, hub_indices, control_indices,
                partition_result, model_template, device):
    """
    RQ2: Compute per-client L2 AUC, cross-edge count, hop distance.
    Returns list of dicts (one per client).
    """
    num_clients = partition_result.num_clients
    cross_matrix = partition_result.cross_edge_matrix
    pm = partition_result.partition_map

    # Hop distances from target
    hop_neighbors = get_k_hop_neighbors_fast(
        full_data.edge_index, target_idx, k=5, num_nodes=full_data.num_nodes
    )
    node_distances = {target_idx: 0}
    for hop, nodes in enumerate(hop_neighbors):
        for n in nodes.tolist():
            if n not in node_distances:
                node_distances[n] = hop + 1

    results = []
    for cid in range(num_clients):
        subgraph = client_subgraphs[cid]
        is_target = (cid == target_client_id)

        # Cross-edge count
        if is_target:
            cross_edge_count = -1
        else:
            cross_edge_count = (
                cross_matrix[target_client_id, cid].item() +
                cross_matrix[cid, target_client_id].item()
            )

        # Min hop distance
        client_nodes = (pm == cid).nonzero(as_tuple=True)[0].tolist()
        min_hop = min(
            (node_distances.get(n, 999) for n in client_nodes), default=999,
        )

        # Compute L2/Conf AUC
        if is_target:
            local_before = snap_before.local_states.get(cid)
            local_after = snap_after.local_states.get(cid)
            if local_before is not None and local_after is not None:
                model_b = _load_model(model_template, local_before, device)
                model_a = _load_model(model_template, local_after, device)
                local_data = subgraph.data.to(device)
                emb_b = get_embeddings(model_b, local_data)
                emb_a = get_embeddings(model_a, local_data)
                _, local_hubs = filter_nodes_in_client(hub_indices, subgraph)
                _, local_ctrls = filter_nodes_in_client(control_indices, subgraph)
                if len(local_hubs) >= 2 and len(local_ctrls) >= 2:
                    l2_auc = _safe_auc(
                        _compute_drifts(emb_b, emb_a, local_hubs, 'l2'),
                        _compute_drifts(emb_b, emb_a, local_ctrls, 'l2'))
                    conf_auc = _safe_auc(
                        _compute_conf_drifts(emb_b, emb_a, local_hubs),
                        _compute_conf_drifts(emb_b, emb_a, local_ctrls))
                else:
                    l2_auc, conf_auc = 0.5, 0.5
            else:
                l2_auc, conf_auc = 0.5, 0.5
        else:
            model_b = _load_model(model_template, snap_before.global_state, device)
            model_a = _load_model(model_template, snap_after.global_state, device)
            local_data = subgraph.data.to(device)
            emb_b = get_embeddings(model_b, local_data)
            emb_a = get_embeddings(model_a, local_data)
            _, local_hubs = filter_nodes_in_client(hub_indices, subgraph)
            _, local_ctrls = filter_nodes_in_client(control_indices, subgraph)
            if len(local_hubs) >= 2 and len(local_ctrls) >= 2:
                l2_auc = _safe_auc(
                    _compute_drifts(emb_b, emb_a, local_hubs, 'l2'),
                    _compute_drifts(emb_b, emb_a, local_ctrls, 'l2'))
                conf_auc = _safe_auc(
                    _compute_conf_drifts(emb_b, emb_a, local_hubs),
                    _compute_conf_drifts(emb_b, emb_a, local_ctrls))
            else:
                l2_auc, conf_auc = 0.5, 0.5

        results.append({
            'client_id': cid,
            'is_target_client': is_target,
            'l2_auc': l2_auc,
            'conf_auc': conf_auc,
            'cross_edge_count': cross_edge_count,
            'min_hop_distance': min_hop,
            'num_client_nodes': len(client_nodes),
        })

    return results


# ========================================================================
# RQ4: Threat model evaluation
# ========================================================================

def collect_rq4(snap_before, snap_after, full_data, client_subgraphs,
                target_idx, target_client_id, hub_indices, control_indices,
                model_template, device, num_clients, trial_idx):
    """
    RQ4: Evaluate TM1-TM4 (+ TM2-Same, TM2-Cross variants).
    Returns list of dicts (one per threat model).
    """
    threat_models = [
        ('TM1_WhiteBox', TM1_WhiteBox()),
        ('TM2_Same', TM2_LocalAuditor(target_client_id)),
        ('TM3_ServerAuditor', TM3_ServerAuditor()),
        ('TM4_BlackBox', TM4_BlackBox()),
    ]

    # TM2-Cross: pick a non-target client as auditor
    other_clients = [c for c in range(num_clients) if c != target_client_id]
    if other_clients:
        auditor_cid = other_clients[trial_idx % len(other_clients)]
        threat_models.append(('TM2_Cross', TM2_LocalAuditor(auditor_cid)))

    results = []
    for tm_name, tm in threat_models:
        tm_result = evaluate_threat_model(
            tm, snap_before, snap_after,
            full_data, client_subgraphs,
            target_idx, target_client_id,
            hub_indices, control_indices,
            model_template, device,
        )
        row = {
            'threat_model': tm_name,
            'l2_auc': tm_result.l2_auc,
            'conf_auc': tm_result.conf_auc,
            'available_levels': ','.join(tm_result.available_levels),
        }
        for k, v in tm_result.detail.items():
            row[f'detail_{k}'] = v
        results.append(row)

    return results


# ========================================================================
# RQ5: multi-level results (also serves RQ3)
# ========================================================================

def collect_rq5(snap_before, snap_after, full_data, client_subgraphs,
                target_idx, target_client_id, hub_indices, control_indices,
                model_template, device, unlearn_time, granularity, method):
    """
    RQ5: Multi-level attack result for one method.
    Returns a single dict.
    """
    attack = multilevel_hub_ripple(
        snap_before, snap_after,
        full_data, client_subgraphs,
        target_idx, target_client_id,
        hub_indices, control_indices,
        model_template, device,
    )
    return {
        'granularity': granularity,
        'method': method,
        'global_l2_auc': attack.global_l2_auc,
        'global_conf_auc': attack.global_conf_auc,
        'global_gap': attack.global_gap,
        'local_l2_auc': attack.local_l2_auc,
        'local_conf_auc': attack.local_conf_auc,
        'local_gap': attack.local_gap,
        'mean_cross_l2_auc': attack.mean_cross_l2_auc,
        'max_cross_l2_auc': attack.max_cross_l2_auc,
        'num_hubs': attack.num_hubs,
        'unlearn_time': unlearn_time,
    }


# ========================================================================
# Main trial loop
# ========================================================================

def run_trial(data, partition_result, client_subgraphs, model_template,
              device, grid, trial_idx, seed=42):
    """
    Single trial: train once, then collect RQ2+RQ4+RQ5 for all methods.

    Returns (rq2_rows, rq4_rows, rq5_rows) or None if invalid.
    """
    trial_seed = seed + trial_idx
    set_seed(trial_seed)

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1
    num_clients = grid['num_clients']

    target_idx, target_client_id, hubs, ctrls = \
        select_target_and_hubs(data, partition_result, seed=trial_seed)

    if len(hubs) < 3 or len(ctrls) < 3:
        return None

    # ---- Train federated model (shared across all methods) ----
    global_model = GCN2Layer(num_features, num_classes).to(device)
    set_seed(trial_seed)
    global_model.reset_parameters()

    clients = []
    for cid, subgraph in client_subgraphs.items():
        clients.append(FederatedClient(cid, copy.deepcopy(subgraph), global_model, device))

    server = FederatedServer(global_model, clients, device)
    server.train(
        num_rounds=grid['num_rounds'],
        local_epochs=grid['local_epochs'],
        lr=0.01, weight_decay=5e-4,
    )
    server.snapshot_state('before')
    snap_before = server.get_snapshot('before')

    rq2_rows = []
    rq4_rows = []
    rq5_rows = []

    # ---- Node-level unlearning (5 methods) ----
    for method in grid['methods']:
        ul_server = clone_server(server)

        t0 = time.time()
        result = do_unlearn(
            ul_server, method, target_idx, target_client_id,
            data, partition_result, num_features, num_classes,
            grid, trial_seed,
        )
        unlearn_time = time.time() - t0

        ul_server = result.server
        ul_server.snapshot_state('after')
        snap_after = ul_server.get_snapshot('after')

        meta = {
            'method': method,
            'target_idx': target_idx,
            'target_client_id': target_client_id,
            'num_hubs': len(hubs),
        }

        # RQ2: per-client leakage
        for row in collect_rq2(
            snap_before, snap_after, data, client_subgraphs,
            target_idx, target_client_id, hubs, ctrls,
            partition_result, model_template, device,
        ):
            row.update(meta)
            rq2_rows.append(row)

        # RQ4: threat models
        for row in collect_rq4(
            snap_before, snap_after, data, client_subgraphs,
            target_idx, target_client_id, hubs, ctrls,
            model_template, device, num_clients, trial_idx,
        ):
            row.update(meta)
            rq4_rows.append(row)

        # RQ5: multi-level (node granularity)
        row = collect_rq5(
            snap_before, snap_after, data, client_subgraphs,
            target_idx, target_client_id, hubs, ctrls,
            model_template, device, unlearn_time, 'node', method,
        )
        row.update(meta)
        rq5_rows.append(row)

    # ---- Client-level unlearning (RQ5 only) ----
    ul_server = clone_server(server)
    t0 = time.time()
    result = fed_client_unlearning(
        ul_server, target_client_id,
        reaggregate_rounds=5,
        local_epochs=grid['local_epochs'],
        lr=0.01, weight_decay=5e-4,
    )
    unlearn_time = time.time() - t0

    ul_server = result.server
    ul_server.snapshot_state('after')
    snap_after = ul_server.get_snapshot('after')

    row = collect_rq5(
        snap_before, snap_after, data, client_subgraphs,
        target_idx, target_client_id, hubs, ctrls,
        model_template, device, unlearn_time, 'client', 'FedClientUnlearning',
    )
    row.update({
        'method': 'FedClientUnlearning',
        'target_idx': target_idx,
        'target_client_id': target_client_id,
        'num_hubs': len(hubs),
    })
    rq5_rows.append(row)

    return rq2_rows, rq4_rows, rq5_rows


# ========================================================================
# Summary printing
# ========================================================================

def print_rq2_summary(df):
    """Print RQ2 analysis tables."""
    print("\n" + "=" * 70)
    print("RQ2 ANALYSIS: Cross-Client Leakage Correlation")
    print("=" * 70)

    cross_df = df[~df['is_target_client']]

    # Table 1: Correlation per dataset × method
    print("\n[Table 1] Pearson r (cross_edge_count vs L2 AUC)")
    print(f"{'Dataset':<15} {'Method':<22} {'r':>8} {'sig':>6} {'N':>6}")
    print("─" * 60)
    for ds in df['dataset'].unique():
        for method in df['method'].unique():
            subset = cross_df[(cross_df['dataset'] == ds) &
                              (cross_df['method'] == method)]
            if len(subset) < 5:
                continue
            edges = subset['cross_edge_count'].values
            aucs = subset['l2_auc'].values
            if np.std(edges) > 0 and np.std(aucs) > 0:
                r = np.corrcoef(edges, aucs)[0, 1]
                sig = "YES" if abs(r) > 0.3 else "weak"
            else:
                r, sig = 0.0, "N/A"
            print(f"{ds:<15} {method:<22} {r:>8.4f} {sig:>6} {len(subset):>6}")

    # Table 2: by hop distance
    print("\n[Table 2] Mean L2 AUC by Hop Distance")
    print(f"{'Hop':>6} {'Mean L2':>10} {'Std':>8} {'N':>6}")
    print("─" * 32)
    for hop in sorted(cross_df['min_hop_distance'].unique()):
        if hop > 10:
            continue
        s = cross_df[cross_df['min_hop_distance'] == hop]
        print(f"{hop:>6} {s['l2_auc'].mean():>10.4f} {s['l2_auc'].std():>8.4f} {len(s):>6}")

    # Table 3: target vs non-target
    target_df = df[df['is_target_client']]
    print("\n[Table 3] Target (Local) vs Non-Target (Cross)")
    print(f"{'Level':<15} {'L2 AUC':>10} {'Conf AUC':>10} {'N':>6}")
    print("─" * 43)
    if len(target_df) > 0:
        print(f"{'Target':<15} {target_df['l2_auc'].mean():>10.4f} "
              f"{target_df['conf_auc'].mean():>10.4f} {len(target_df):>6}")
    if len(cross_df) > 0:
        print(f"{'Non-Target':<15} {cross_df['l2_auc'].mean():>10.4f} "
              f"{cross_df['conf_auc'].mean():>10.4f} {len(cross_df):>6}")

    # Overall
    if len(cross_df) > 5:
        edges = cross_df['cross_edge_count'].values
        aucs = cross_df['l2_auc'].values
        if np.std(edges) > 0 and np.std(aucs) > 0:
            r = np.corrcoef(edges, aucs)[0, 1]
        else:
            r = 0.0
        print(f"\nOverall r = {r:.4f} → "
              f"{'CONFIRMED' if abs(r) > 0.3 else 'WEAK'}")


def print_rq4_summary(df):
    """Print RQ4 analysis tables."""
    print("\n" + "=" * 70)
    print("RQ4 ANALYSIS: Threat Model Comparison")
    print("=" * 70)

    tm_order = ['TM1_WhiteBox', 'TM2_Same', 'TM3_ServerAuditor',
                'TM2_Cross', 'TM4_BlackBox']

    # Table 1: overall
    print("\n[Table 1] Mean AUC by Threat Model (all datasets, all methods)")
    print(f"{'Threat Model':<20} {'L2 AUC':>10} {'Conf AUC':>10} {'Gap':>8} {'N':>6}")
    print("─" * 56)
    for tm in tm_order:
        s = df[df['threat_model'] == tm]
        if len(s) == 0:
            continue
        l2 = s['l2_auc'].mean()
        conf = s['conf_auc'].mean()
        print(f"{tm:<20} {l2:>10.4f} {conf:>10.4f} {l2 - conf:>+8.4f} {len(s):>6}")

    # Table 2: per dataset
    print("\n[Table 2] L2 AUC by Threat Model × Dataset")
    for ds in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds]
        print(f"\n  {ds}:")
        print(f"  {'TM':<20} {'L2':>8} {'Conf':>8}")
        print(f"  {'─' * 38}")
        for tm in tm_order:
            s = ds_df[ds_df['threat_model'] == tm]
            if len(s) == 0:
                continue
            print(f"  {tm:<20} {s['l2_auc'].mean():>8.4f} {s['conf_auc'].mean():>8.4f}")

    # Table 3: per method
    print("\n[Table 3] L2 AUC by Threat Model × Method")
    for method in df['method'].unique():
        m_df = df[df['method'] == method]
        print(f"\n  {method}:")
        print(f"  {'TM':<20} {'L2':>8} {'Conf':>8}")
        print(f"  {'─' * 38}")
        for tm in tm_order:
            s = m_df[m_df['threat_model'] == tm]
            if len(s) == 0:
                continue
            print(f"  {tm:<20} {s['l2_auc'].mean():>8.4f} {s['conf_auc'].mean():>8.4f}")

    # Verification
    tm_means = {}
    for tm in tm_order:
        s = df[df['threat_model'] == tm]
        if len(s) > 0:
            tm_means[tm] = s['l2_auc'].mean()

    if 'TM4_BlackBox' in tm_means:
        print(f"\n  TM4 blind to L2: {'CONFIRMED' if abs(tm_means['TM4_BlackBox'] - 0.5) < 0.05 else 'UNEXPECTED'} "
              f"(L2={tm_means['TM4_BlackBox']:.4f})")


def print_rq5_summary(df):
    """Print RQ5 analysis tables."""
    print("\n" + "=" * 70)
    print("RQ5 ANALYSIS: Node-Level vs Client-Level Unlearning")
    print("=" * 70)

    # Table 1: overall
    print("\n[Table 1] Mean AUC by Granularity (all datasets)")
    print(f"{'Gran.':<10} {'Method':<22} {'G-L2':>8} {'G-Conf':>8} "
          f"{'Gap':>8} {'L-L2':>8} {'X-L2':>8} {'Time':>8}")
    print("─" * 86)
    for gran in ['node', 'client']:
        g_df = df[df['granularity'] == gran]
        for method in g_df['method'].unique():
            m = g_df[g_df['method'] == method]
            print(f"{gran:<10} {method:<22} "
                  f"{m['global_l2_auc'].mean():>8.4f} "
                  f"{m['global_conf_auc'].mean():>8.4f} "
                  f"{m['global_gap'].mean():>+8.4f} "
                  f"{m['local_l2_auc'].mean():>8.4f} "
                  f"{m['mean_cross_l2_auc'].mean():>8.4f} "
                  f"{m['unlearn_time'].mean():>7.2f}s")

    # Table 2: paired comparison
    node_df = df[df['granularity'] == 'node']
    client_df = df[df['granularity'] == 'client']
    if len(node_df) > 0 and len(client_df) > 0:
        print("\n[Table 2] Paired: Node-Level (avg) vs Client-Level")
        print(f"  {'Metric':<20} {'Node':>10} {'Client':>10} {'Delta':>10}")
        print(f"  {'─' * 52}")
        for col in ['global_l2_auc', 'global_conf_auc', 'local_l2_auc', 'mean_cross_l2_auc']:
            n = node_df[col].mean()
            c = client_df[col].mean()
            print(f"  {col:<20} {n:>10.4f} {c:>10.4f} {c - n:>+10.4f}")

    # Verification
    n_l2 = node_df['global_l2_auc'].mean() if len(node_df) > 0 else 0.5
    c_l2 = client_df['global_l2_auc'].mean() if len(client_df) > 0 else 0.5
    print(f"\n  Node L2={n_l2:.4f}, Client L2={c_l2:.4f}, Delta={c_l2 - n_l2:+.4f}")
    print(f"  → {'CONFIRMED' if c_l2 < n_l2 else 'UNEXPECTED'}: "
          f"Client-level {'better' if c_l2 < n_l2 else 'worse'} privacy")
    if c_l2 > 0.55:
        print(f"  → Residual leakage persists (Client L2={c_l2:.4f} > 0.55)")


# ========================================================================
# Main
# ========================================================================

def main():
    args = parse_args()
    device = get_device()

    if args.quick:
        grid = {
            'datasets': ['Cora'],
            'methods': ['FedRetrain', 'FedGNNDelete'],
            'num_clients': 5,
            'num_rounds': 30,
            'local_epochs': 3,
            'epochs': 100,
        }
    else:
        grid = {
            'datasets': ['Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel'],
            'methods': ['FedRetrain', 'FedGNNDelete', 'FedGIF',
                        'FedGraphEraser-BEKM', 'FedGraphEraser-BLPA'],
            'num_clients': 5,
            'num_rounds': 50,
            'local_epochs': 5,
            'epochs': 200,
        }

    print("=" * 70)
    print("Unified Supplementary Experiment: RQ2 + RQ4 + RQ5")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Trials: {args.trials}")
    print(f"Datasets: {grid['datasets']}")
    print(f"Methods: {grid['methods']} ({len(grid['methods'])} methods)")
    print(f"+ Client-level: FedClientUnlearning")
    total = len(grid['datasets']) * args.trials
    total_unlearns = total * (len(grid['methods']) + 1)  # +1 for client-level
    print(f"Total trials: {total} (each → {len(grid['methods'])+1} unlearnings = {total_unlearns})")
    print("=" * 70)

    all_rq2 = []
    all_rq4 = []
    all_rq5 = []

    for ds_idx, dataset_name in enumerate(grid['datasets']):
        print(f"\n{'━' * 70}")
        print(f"Dataset: {dataset_name} [{ds_idx + 1}/{len(grid['datasets'])}]")
        print(f"{'━' * 70}")

        try:
            data, homophily = load_dataset(dataset_name, device)
        except Exception as e:
            print(f"  SKIP: Failed to load {dataset_name}: {e}")
            continue

        num_features = data.num_features
        num_classes = int(data.y.max().item()) + 1
        model_template = GCN2Layer(num_features, num_classes).to(device)

        partition_result = partition_graph(
            data, grid['num_clients'], method='metis', seed=args.seed,
        )
        print(f"  Partition: metis, K={grid['num_clients']}, "
              f"cross_ratio={partition_result.cross_edge_ratio:.3f}")

        client_subgraphs = {}
        for cid in range(grid['num_clients']):
            nodes = (partition_result.partition_map == cid).nonzero(as_tuple=True)[0]
            client_subgraphs[cid] = build_client_subgraph(
                data, nodes, data.edge_index, cid
            )

        t_ds_start = time.time()
        n_valid = 0

        for trial in range(args.trials):
            result = run_trial(
                data, partition_result, client_subgraphs, model_template,
                device, grid, trial_idx=trial, seed=args.seed,
            )
            if result is None:
                continue

            rq2_rows, rq4_rows, rq5_rows = result
            n_valid += 1

            common = {
                'dataset': dataset_name,
                'trial': trial,
                'homophily': homophily,
                'cross_edge_ratio': partition_result.cross_edge_ratio,
            }

            for row in rq2_rows:
                row.update(common)
                all_rq2.append(row)

            for row in rq4_rows:
                row.update(common)
                all_rq4.append(row)

            for row in rq5_rows:
                row.update(common)
                all_rq5.append(row)

            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t_ds_start
                print(f"  Trial {trial + 1}/{args.trials} "
                      f"({n_valid} valid, {elapsed:.0f}s)")

        ds_elapsed = time.time() - t_ds_start
        print(f"  Done: {n_valid} valid trials, {ds_elapsed:.0f}s")

    # ================================================================
    # Save results
    # ================================================================
    output_dir = 'results/federated/tables'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    paths = {}
    for name, rows in [('rq2_cross_client', all_rq2),
                        ('rq4_threat_model', all_rq4),
                        ('rq5_granularity', all_rq5)]:
        if not rows:
            continue
        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, f'{name}_{timestamp}.csv')
        df.to_csv(path, index=False)
        paths[name] = path
        print(f"\n{name}: {len(df)} rows → {path}")

    # ================================================================
    # Print summaries
    # ================================================================
    if all_rq2:
        print_rq2_summary(pd.DataFrame(all_rq2))
    if all_rq4:
        print_rq4_summary(pd.DataFrame(all_rq4))
    if all_rq5:
        print_rq5_summary(pd.DataFrame(all_rq5))

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == '__main__':
    main()
