"""
RQ5: Unlearning Granularity Comparison — Node-Level vs Client-Level

Compares two unlearning granularities:
1. Node-level: Remove a single target node (FedGNNDelete, FedGIF, FedGraphEraser)
2. Client-level: Entire client departs the federation (FedClientUnlearning)

For each trial:
- Train federated model
- Perform node-level unlearning → measure multi-level Hub-Ripple AUC
- Perform client-level unlearning (same target's client) → measure AUC
- Compare: client-level should erase more but may cause cross-client leakage

Expected: Client-level achieves lower L2 AUC (better privacy) than node-level,
but both still show L2 AUC > 0.5 due to cross-client parameter propagation.

Output columns:
  dataset, granularity, method, trial, global_l2_auc, global_conf_auc,
  global_gap, local_l2_auc, local_conf_auc, mean_cross_l2_auc,
  max_cross_l2_auc, unlearn_time

Usage:
    python experiments/federated/run_fed_granularity.py --trials 100
    python experiments/federated/run_fed_granularity.py --quick --trials 20
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
)
from src.models.gcn import GCN2Layer
from src.federated.subgraph import build_client_subgraph
from src.federated.data_partition import partition_graph
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.unlearning import (
    fed_retrain, fed_gnndelete, fed_gif, fed_graph_eraser, fed_client_unlearning,
)
from src.federated.attacks.hub_ripple_federated import multilevel_hub_ripple


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


def create_trained_server(data, client_subgraphs, device, grid, trial_seed):
    """Create and train a fresh server, return (server, global_acc)."""
    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

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

    global_acc = server.evaluate_global(data)
    return server, global_acc


def clone_server(server):
    """Deep copy a server for independent unlearning."""
    ul_clients = []
    for cid in server.client_ids:
        ul_clients.append(FederatedClient(
            cid, copy.deepcopy(server.clients[cid].subgraph),
            server.clients[cid].model, server.device,
        ))
    return FederatedServer(copy.deepcopy(server.global_model), ul_clients, server.device)


def run_trial(data, partition_result, client_subgraphs, model_template,
              device, trial_idx, grid, seed=42):
    """
    Run single trial: train once, then perform both node-level and
    client-level unlearning, returning results for all methods.
    """
    trial_seed = seed + trial_idx
    set_seed(trial_seed)

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

    target_idx, target_client_id, hubs, ctrls = \
        select_target_and_hubs(data, partition_result, seed=trial_seed)

    if len(hubs) < 3 or len(ctrls) < 3:
        return None

    # Train shared server
    server, global_acc = create_trained_server(
        data, client_subgraphs, device, grid, trial_seed
    )
    server.snapshot_state('before')
    snap_before = server.get_snapshot('before')

    results = []

    # ===== Node-level unlearning methods =====
    node_methods = grid['node_methods']

    for method in node_methods:
        ul_server = clone_server(server)

        t0 = time.time()
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
            continue
        unlearn_time = time.time() - t0

        ul_server = result.server
        ul_server.snapshot_state('after')
        snap_after = ul_server.get_snapshot('after')

        attack = multilevel_hub_ripple(
            snap_before, snap_after,
            data, client_subgraphs,
            target_idx, target_client_id,
            hubs, ctrls,
            model_template, device,
        )

        results.append({
            'granularity': 'node',
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
            'global_acc_before': global_acc,
            'target_idx': target_idx,
            'target_client_id': target_client_id,
        })

    # ===== Client-level unlearning =====
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

    # For client-level unlearning, the target client is gone.
    # We evaluate using the remaining client subgraphs.
    # The global model was re-aggregated from K-1 clients.
    # We still use the original client_subgraphs for hub/control evaluation.
    attack = multilevel_hub_ripple(
        snap_before, snap_after,
        data, client_subgraphs,
        target_idx, target_client_id,
        hubs, ctrls,
        model_template, device,
    )

    results.append({
        'granularity': 'client',
        'method': 'FedClientUnlearning',
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
        'global_acc_before': global_acc,
        'target_idx': target_idx,
        'target_client_id': target_client_id,
    })

    return results


def main():
    args = parse_args()
    device = get_device()

    if args.quick:
        grid = {
            'datasets': ['Cora'],
            'node_methods': ['FedRetrain', 'FedGNNDelete'],
            'num_clients': 5,
            'num_rounds': 30,
            'local_epochs': 3,
            'epochs': 100,
        }
    else:
        grid = {
            'datasets': ['Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel'],
            'node_methods': ['FedRetrain', 'FedGNNDelete', 'FedGIF',
                             'FedGraphEraser-BEKM', 'FedGraphEraser-BLPA'],
            'num_clients': 5,
            'num_rounds': 50,
            'local_epochs': 5,
            'epochs': 200,
        }

    print("=" * 70)
    print("RQ5: Unlearning Granularity — Node-Level vs Client-Level")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Trials: {args.trials}")
    print(f"Datasets: {grid['datasets']}")
    print(f"Node methods: {grid['node_methods']} ({len(grid['node_methods'])} methods)")
    print(f"Client method: FedClientUnlearning")
    total = len(grid['datasets']) * args.trials
    print(f"Total trial runs: {total}")
    print("=" * 70)

    all_results = []

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

        t_start = time.time()
        n_valid = 0

        for trial in range(args.trials):
            trial_results = run_trial(
                data, partition_result, client_subgraphs,
                model_template, device,
                trial_idx=trial, grid=grid, seed=args.seed,
            )
            if trial_results is None:
                continue

            n_valid += 1
            for row in trial_results:
                row.update({
                    'dataset': dataset_name,
                    'trial': trial,
                    'homophily': homophily,
                })
                all_results.append(row)

            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"    Trial {trial + 1}/{args.trials} "
                      f"({n_valid} valid, {elapsed:.1f}s)")

        elapsed = time.time() - t_start
        print(f"  Total: {n_valid} valid trials, {elapsed:.1f}s")

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
        output_path = os.path.join(output_dir, f'rq5_granularity_{timestamp}.csv')

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # ================================================================
    # Summary analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("RQ5 ANALYSIS: Node-Level vs Client-Level Unlearning")
    print("=" * 70)

    # Table 1: Overall comparison
    print("\n[Table 1] Mean AUC by Granularity (all datasets)")
    print(f"{'Granularity':<12} {'Method':<22} {'G-L2':>8} {'G-Conf':>8} "
          f"{'Gap':>8} {'L-L2':>8} {'X-L2':>8} {'Time':>8}")
    print("─" * 82)

    for granularity in ['node', 'client']:
        g_df = df[df['granularity'] == granularity]
        for method in g_df['method'].unique():
            m_df = g_df[g_df['method'] == method]
            print(f"{granularity:<12} {method:<22} "
                  f"{m_df['global_l2_auc'].mean():>8.4f} "
                  f"{m_df['global_conf_auc'].mean():>8.4f} "
                  f"{m_df['global_gap'].mean():>+8.4f} "
                  f"{m_df['local_l2_auc'].mean():>8.4f} "
                  f"{m_df['mean_cross_l2_auc'].mean():>8.4f} "
                  f"{m_df['unlearn_time'].mean():>8.2f}s")

    # Table 2: Per-dataset comparison
    print("\n[Table 2] Global L2 AUC by Granularity × Dataset")
    for dataset_name in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset_name]
        print(f"\n  {dataset_name}:")
        print(f"  {'Granularity':<12} {'Method':<22} {'G-L2':>8} {'L-L2':>8} {'X-L2':>8}")
        print(f"  {'─' * 56}")
        for granularity in ['node', 'client']:
            g_df = ds_df[ds_df['granularity'] == granularity]
            for method in g_df['method'].unique():
                m_df = g_df[g_df['method'] == method]
                print(f"  {granularity:<12} {method:<22} "
                      f"{m_df['global_l2_auc'].mean():>8.4f} "
                      f"{m_df['local_l2_auc'].mean():>8.4f} "
                      f"{m_df['mean_cross_l2_auc'].mean():>8.4f}")

    # Table 3: Paired comparison (node best vs client)
    print("\n[Table 3] Paired Comparison: Best Node-Level vs Client-Level")
    node_df = df[df['granularity'] == 'node']
    client_df = df[df['granularity'] == 'client']

    if len(node_df) > 0 and len(client_df) > 0:
        # Average across node methods per trial
        node_avg = node_df.groupby(['dataset', 'trial']).agg({
            'global_l2_auc': 'mean',
            'global_conf_auc': 'mean',
            'local_l2_auc': 'mean',
            'mean_cross_l2_auc': 'mean',
        }).reset_index()

        client_avg = client_df.groupby(['dataset', 'trial']).agg({
            'global_l2_auc': 'mean',
            'global_conf_auc': 'mean',
            'local_l2_auc': 'mean',
            'mean_cross_l2_auc': 'mean',
        }).reset_index()

        print(f"  {'Metric':<20} {'Node (avg)':>12} {'Client':>12} {'Delta':>10}")
        print(f"  {'─' * 56}")

        for metric in ['global_l2_auc', 'global_conf_auc', 'local_l2_auc', 'mean_cross_l2_auc']:
            n_val = node_avg[metric].mean()
            c_val = client_avg[metric].mean()
            delta = c_val - n_val
            print(f"  {metric:<20} {n_val:>12.4f} {c_val:>12.4f} {delta:>+10.4f}")

    # Verification
    print("\n" + "=" * 70)
    print("RQ5 Verification")
    print("=" * 70)

    node_l2 = node_df['global_l2_auc'].mean() if len(node_df) > 0 else 0.5
    client_l2 = client_df['global_l2_auc'].mean() if len(client_df) > 0 else 0.5

    print(f"  Node-level avg Global L2 AUC:   {node_l2:.4f}")
    print(f"  Client-level avg Global L2 AUC: {client_l2:.4f}")
    print(f"  Delta: {client_l2 - node_l2:+.4f}")

    if client_l2 < node_l2:
        print(f"  → CONFIRMED: Client-level unlearning provides better privacy")
    else:
        print(f"  → UNEXPECTED: Client-level does not provide better privacy")

    # Check if both still > 0.5
    if client_l2 > 0.55:
        print(f"  → Client-level still leaks (L2 AUC={client_l2:.4f} > 0.55): "
              f"cross-client leakage persists through parameter history")
    else:
        print(f"  → Client-level effective (L2 AUC={client_l2:.4f} ≈ 0.5)")


if __name__ == '__main__':
    main()
