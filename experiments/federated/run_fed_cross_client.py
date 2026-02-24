"""
RQ2: Cross-Client Leakage Correlation Experiment

Quantifies how unlearning information propagates across client boundaries
and correlates leakage with cross-client edge connectivity.

For each trial:
1. Train federated model
2. Perform unlearning on target client
3. Measure per-client L2 AUC (leakage to each non-target client)
4. Record cross-edge count between target client and each other client
5. Compute correlation (Pearson r) between cross-edges and L2 AUC

Also performs hop-distance analysis: how leakage decays with graph distance.

Output columns:
  dataset, method, trial, client_id, l2_auc, conf_auc,
  cross_edge_count, min_hop_distance, is_target_client

Usage:
    python experiments/federated/run_fed_cross_client.py --trials 100
    python experiments/federated/run_fed_cross_client.py --quick --trials 20
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
from src.federated.unlearning import fed_retrain, fed_gnndelete, fed_gif, fed_graph_eraser
from src.federated.attacks.hub_ripple_federated import (
    multilevel_hub_ripple, _load_model, _compute_drifts,
    _compute_conf_drifts, _safe_auc,
)
from src.models.training import get_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
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


def compute_per_client_leakage(
    snap_before, snap_after, full_data, client_subgraphs,
    target_idx, target_client_id,
    hub_indices, control_indices,
    partition_result, model_template, device,
):
    """
    Compute per-client L2 AUC, cross-edge counts, and hop distances.

    Returns a list of dicts, one per client.
    """
    num_clients = partition_result.num_clients
    cross_matrix = partition_result.cross_edge_matrix
    results = []

    # Compute hop distances from target
    hop_neighbors = get_k_hop_neighbors_fast(
        full_data.edge_index, target_idx, k=5, num_nodes=full_data.num_nodes
    )
    node_distances = {target_idx: 0}
    for hop, nodes in enumerate(hop_neighbors):
        for n in nodes.tolist():
            if n not in node_distances:
                node_distances[n] = hop + 1

    pm = partition_result.partition_map

    for cid in range(num_clients):
        subgraph = client_subgraphs[cid]
        is_target = (cid == target_client_id)

        # Cross-edge count to target client
        if is_target:
            cross_edge_count = -1  # N/A for target client itself
        else:
            cross_edge_count = (
                cross_matrix[target_client_id, cid].item() +
                cross_matrix[cid, target_client_id].item()
            )

        # Min hop distance from target to this client
        client_nodes = (pm == cid).nonzero(as_tuple=True)[0].tolist()
        min_hop = min(
            (node_distances.get(n, 999) for n in client_nodes),
            default=999,
        )

        # Compute L2 and Conf AUC for this client
        if is_target:
            # Local level: use local model
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
                    hub_l2 = _compute_drifts(emb_b, emb_a, local_hubs, 'l2')
                    ctrl_l2 = _compute_drifts(emb_b, emb_a, local_ctrls, 'l2')
                    l2_auc = _safe_auc(hub_l2, ctrl_l2)

                    hub_conf = _compute_conf_drifts(emb_b, emb_a, local_hubs)
                    ctrl_conf = _compute_conf_drifts(emb_b, emb_a, local_ctrls)
                    conf_auc = _safe_auc(hub_conf, ctrl_conf)
                else:
                    l2_auc, conf_auc = 0.5, 0.5
            else:
                l2_auc, conf_auc = 0.5, 0.5
        else:
            # Cross-client level: use global model on other client's subgraph
            model_b = _load_model(model_template, snap_before.global_state, device)
            model_a = _load_model(model_template, snap_after.global_state, device)
            local_data = subgraph.data.to(device)
            emb_b = get_embeddings(model_b, local_data)
            emb_a = get_embeddings(model_a, local_data)

            _, local_hubs = filter_nodes_in_client(hub_indices, subgraph)
            _, local_ctrls = filter_nodes_in_client(control_indices, subgraph)

            if len(local_hubs) >= 2 and len(local_ctrls) >= 2:
                hub_l2 = _compute_drifts(emb_b, emb_a, local_hubs, 'l2')
                ctrl_l2 = _compute_drifts(emb_b, emb_a, local_ctrls, 'l2')
                l2_auc = _safe_auc(hub_l2, ctrl_l2)

                hub_conf = _compute_conf_drifts(emb_b, emb_a, local_hubs)
                ctrl_conf = _compute_conf_drifts(emb_b, emb_a, local_ctrls)
                conf_auc = _safe_auc(hub_conf, ctrl_conf)
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


def run_trial(data, partition_result, client_subgraphs, model_template,
              device, method, trial_idx, grid, seed=42):
    """Run a single trial. Returns list of per-client result dicts."""
    trial_seed = seed + trial_idx
    set_seed(trial_seed)

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

    target_idx, target_client_id, hubs, ctrls = \
        select_target_and_hubs(data, partition_result, seed=trial_seed)

    if len(hubs) < 3 or len(ctrls) < 3:
        return None

    # Fresh model + clients
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

    # Deep copy for unlearning
    ul_clients = []
    for cid in server.client_ids:
        ul_clients.append(FederatedClient(
            cid, copy.deepcopy(server.clients[cid].subgraph),
            server.clients[cid].model, device,
        ))
    ul_server = FederatedServer(copy.deepcopy(server.global_model), ul_clients, device)

    # Unlearn
    if method == 'FedRetrain':
        result = fed_retrain(
            ul_server, target_idx, target_client_id,
            data, partition_result, num_features, num_classes,
            num_rounds=grid['num_rounds'], local_epochs=grid['local_epochs'],
            seed=trial_seed + 1000,
        )
    elif method == 'FedGNNDelete':
        result = fed_gnndelete(
            ul_server, target_idx, target_client_id,
            num_features, num_classes,
            reaggregate_rounds=5, local_epochs=grid['local_epochs'],
        )
    elif method == 'FedGIF':
        result = fed_gif(
            ul_server, target_idx, target_client_id,
            num_features, num_classes,
            reaggregate_rounds=5, local_epochs=grid['local_epochs'],
        )
    elif method == 'FedGraphEraser-BEKM':
        result = fed_graph_eraser(
            ul_server, target_idx, target_client_id,
            num_features, num_classes,
            reaggregate_rounds=5, local_epochs=grid['local_epochs'],
            partition_strategy='bekm',
            epochs=grid['epochs'], seed=trial_seed,
        )
    elif method == 'FedGraphEraser-BLPA':
        result = fed_graph_eraser(
            ul_server, target_idx, target_client_id,
            num_features, num_classes,
            reaggregate_rounds=5, local_epochs=grid['local_epochs'],
            partition_strategy='blpa',
            epochs=grid['epochs'], seed=trial_seed,
        )
    else:
        return None

    ul_server = result.server
    ul_server.snapshot_state('after')

    snap_before = server.get_snapshot('before')
    snap_after = ul_server.get_snapshot('after')

    # Per-client leakage analysis
    per_client = compute_per_client_leakage(
        snap_before, snap_after, data, client_subgraphs,
        target_idx, target_client_id,
        hubs, ctrls,
        partition_result, model_template, device,
    )

    # Add metadata to each per-client result
    for row in per_client:
        row['target_idx'] = target_idx
        row['target_client_id'] = target_client_id
        row['num_hubs'] = len(hubs)

    return per_client


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
    print("RQ2: Cross-Client Leakage Correlation Experiment")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Trials: {args.trials}")
    print(f"Datasets: {grid['datasets']}")
    print(f"Methods: {grid['methods']}")
    total = len(grid['datasets']) * len(grid['methods']) * args.trials
    print(f"Total trial runs: {total}")
    print("=" * 70)

    all_results = []
    config_idx = 0
    total_configs = len(grid['datasets']) * len(grid['methods'])

    for dataset_name in grid['datasets']:
        print(f"\n{'━' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'━' * 70}")

        try:
            data, homophily = load_dataset(dataset_name, device)
        except Exception as e:
            print(f"  SKIP: Failed to load {dataset_name}: {e}")
            continue

        num_features = data.num_features
        num_classes = int(data.y.max().item()) + 1
        model_template = GCN2Layer(num_features, num_classes).to(device)

        # Partition (metis only, fixed K=num_clients)
        partition_result = partition_graph(
            data, grid['num_clients'], method='metis', seed=args.seed,
        )
        print(f"  Partition: metis, K={grid['num_clients']}, "
              f"cross_ratio={partition_result.cross_edge_ratio:.3f}")

        # Build subgraphs
        client_subgraphs = {}
        for cid in range(grid['num_clients']):
            nodes = (partition_result.partition_map == cid).nonzero(as_tuple=True)[0]
            client_subgraphs[cid] = build_client_subgraph(
                data, nodes, data.edge_index, cid
            )

        for method in grid['methods']:
            config_idx += 1
            print(f"\n  [{config_idx}/{total_configs}] {method}")

            t_start = time.time()
            n_valid = 0

            for trial in range(args.trials):
                per_client = run_trial(
                    data, partition_result, client_subgraphs,
                    model_template, device, method,
                    trial_idx=trial, grid=grid, seed=args.seed,
                )
                if per_client is None:
                    continue

                n_valid += 1
                for row in per_client:
                    row.update({
                        'dataset': dataset_name,
                        'method': method,
                        'trial': trial,
                        'homophily': homophily,
                        'cross_edge_ratio': partition_result.cross_edge_ratio,
                    })
                    all_results.append(row)

            elapsed = time.time() - t_start
            print(f"    {n_valid} valid trials, {elapsed:.1f}s")

            # Quick summary: correlation for this config
            config_df = pd.DataFrame([r for r in all_results
                                       if r['dataset'] == dataset_name
                                       and r['method'] == method
                                       and not r['is_target_client']])
            if len(config_df) > 5:
                edges = config_df['cross_edge_count'].values
                aucs = config_df['l2_auc'].values
                if np.std(edges) > 0 and np.std(aucs) > 0:
                    r = np.corrcoef(edges, aucs)[0, 1]
                    print(f"    Cross-edge ↔ L2 AUC correlation: r={r:.4f}")

    # Save results
    if not all_results:
        print("\nNo results collected!")
        return

    df = pd.DataFrame(all_results)

    output_dir = 'results/federated/tables'
    os.makedirs(output_dir, exist_ok=True)

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'rq2_cross_client_{timestamp}.csv')

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # ================================================================
    # Summary analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("RQ2 ANALYSIS: Cross-Client Leakage Correlation")
    print("=" * 70)

    # Filter to non-target clients only
    cross_df = df[~df['is_target_client']]

    # Table 1: Correlation per dataset × method
    print("\n[Table 1] Pearson r (cross_edge_count vs L2 AUC)")
    print(f"{'Dataset':<15} {'Method':<22} {'r':>8} {'p_significant':>14} {'N':>6}")
    print("─" * 65)
    for dataset_name in df['dataset'].unique():
        for method in df['method'].unique():
            subset = cross_df[(cross_df['dataset'] == dataset_name) &
                              (cross_df['method'] == method)]
            if len(subset) < 5:
                continue
            edges = subset['cross_edge_count'].values
            aucs = subset['l2_auc'].values
            if np.std(edges) > 0 and np.std(aucs) > 0:
                r = np.corrcoef(edges, aucs)[0, 1]
                significant = "YES" if abs(r) > 0.3 else "weak"
            else:
                r = 0.0
                significant = "N/A"
            print(f"{dataset_name:<15} {method:<22} {r:>8.4f} {significant:>14} {len(subset):>6}")

    # Table 2: Leakage by hop distance
    print("\n[Table 2] Mean L2 AUC by Hop Distance from Target")
    print(f"{'Hop Distance':>12} {'Mean L2 AUC':>12} {'Std':>8} {'N':>6}")
    print("─" * 40)
    for hop in sorted(cross_df['min_hop_distance'].unique()):
        if hop > 10:
            continue
        subset = cross_df[cross_df['min_hop_distance'] == hop]
        print(f"{hop:>12} {subset['l2_auc'].mean():>12.4f} "
              f"{subset['l2_auc'].std():>8.4f} {len(subset):>6}")

    # Table 3: Target client vs non-target client
    target_df = df[df['is_target_client']]
    print("\n[Table 3] Target Client (Local) vs Non-Target Clients (Cross)")
    print(f"{'Level':<15} {'Mean L2 AUC':>12} {'Mean Conf AUC':>14} {'N':>6}")
    print("─" * 50)
    if len(target_df) > 0:
        print(f"{'Target (Local)':<15} {target_df['l2_auc'].mean():>12.4f} "
              f"{target_df['conf_auc'].mean():>14.4f} {len(target_df):>6}")
    if len(cross_df) > 0:
        print(f"{'Non-Target':<15} {cross_df['l2_auc'].mean():>12.4f} "
              f"{cross_df['conf_auc'].mean():>14.4f} {len(cross_df):>6}")

    # Table 4: Leakage threshold (minimum cross-edges for AUC > 0.55)
    print("\n[Table 4] Leakage Detection Threshold")
    for method in cross_df['method'].unique():
        method_df = cross_df[cross_df['method'] == method]
        detectable = method_df[method_df['l2_auc'] > 0.55]
        if len(detectable) > 0:
            min_edges = detectable['cross_edge_count'].min()
            frac = len(detectable) / len(method_df) * 100
            print(f"  {method}: min_cross_edges={min_edges}, "
                  f"detectable={frac:.1f}% of non-target clients")
        else:
            print(f"  {method}: no detectable leakage (all AUC <= 0.55)")

    # Overall summary
    print("\n" + "=" * 70)
    overall_r = 0.0
    if len(cross_df) > 5:
        edges = cross_df['cross_edge_count'].values
        aucs = cross_df['l2_auc'].values
        if np.std(edges) > 0 and np.std(aucs) > 0:
            overall_r = np.corrcoef(edges, aucs)[0, 1]
    print(f"Overall correlation (cross-edges ↔ L2 AUC): r = {overall_r:.4f}")
    print(f"  → {'CONFIRMED' if abs(overall_r) > 0.3 else 'WEAK'}: "
          f"Cross-client leakage correlates with edge connectivity")


if __name__ == '__main__':
    main()
