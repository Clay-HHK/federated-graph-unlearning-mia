"""
Federated Graph Unlearning Privacy Audit — Main Experiment

Full experiment matrix:
  5 datasets × 2 partitions × 3 client counts × 5 methods × 50 trials

Methods: FedRetrain, FedGNNDelete, FedGIF, FedGraphEraser-BEKM, FedGraphEraser-BLPA

Supports --quick mode for faster iteration.

Usage:
    # Standard run (~30-60min CPU)
    python experiments/federated/run_fed_main.py --trials 50

    # Quick run (~10min CPU)
    python experiments/federated/run_fed_main.py --quick --trials 20

    # Full matrix (~8-12h GPU)
    python experiments/federated/run_fed_main.py --full --trials 50
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
from src.models.training import evaluate_model
from src.federated.subgraph import build_client_subgraph
from src.federated.data_partition import partition_graph
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.unlearning import fed_retrain, fed_gnndelete, fed_gif, fed_graph_eraser
from src.federated.attacks.hub_ripple_federated import multilevel_hub_ripple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: Cora only, 3 clients, 2 methods')
    parser.add_argument('--full', action='store_true',
                        help='Full matrix: all datasets, partitions, client counts')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


def get_experiment_grid(args):
    """Build experiment grid based on mode."""
    all_methods = ['FedRetrain', 'FedGNNDelete', 'FedGIF',
                   'FedGraphEraser-BEKM', 'FedGraphEraser-BLPA']
    if args.quick:
        return {
            'datasets': ['Cora', 'CiteSeer'],
            'partition_methods': ['metis'],
            'client_counts': [5],
            'methods': ['FedRetrain', 'FedGNNDelete'],
            'num_rounds': 30,
            'local_epochs': 3,
            'epochs': 100,
        }
    elif args.full:
        return {
            'datasets': ['Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel'],
            'partition_methods': ['metis', 'random'],
            'client_counts': [3, 5, 10],
            'methods': all_methods,
            'num_rounds': 50,
            'local_epochs': 5,
            'epochs': 200,
        }
    else:
        # Standard: balanced between speed and coverage
        return {
            'datasets': ['Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel'],
            'partition_methods': ['metis'],
            'client_counts': [5],
            'methods': all_methods,
            'num_rounds': 50,
            'local_epochs': 5,
            'epochs': 200,
        }


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


def run_trial(data, partition_result, client_subgraphs, model_template,
              device, method, trial_idx, grid, seed=42):
    """Run single trial. Returns dict of results or None."""
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

    # Evaluate before unlearning
    global_acc_before = server.evaluate_global(data)

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
        return None
    unlearn_time = time.time() - t0

    ul_server = result.server
    ul_server.snapshot_state('after')

    # Attack
    snap_before = server.get_snapshot('before')
    snap_after = ul_server.get_snapshot('after')

    attack = multilevel_hub_ripple(
        snap_before, snap_after,
        data, client_subgraphs,
        target_idx, target_client_id,
        hubs, ctrls,
        model_template, device,
    )

    return {
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
        'global_acc_before': global_acc_before,
    }


def main():
    args = parse_args()
    grid = get_experiment_grid(args)
    device = get_device()

    print("=" * 70)
    print("Federated Graph Unlearning — Main Experiment")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Trials per config: {args.trials}")
    print(f"Datasets: {grid['datasets']}")
    print(f"Partitions: {grid['partition_methods']}")
    print(f"Clients: {grid['client_counts']}")
    print(f"Methods: {grid['methods']}")
    total = (len(grid['datasets']) * len(grid['partition_methods']) *
             len(grid['client_counts']) * len(grid['methods']) * args.trials)
    print(f"Total trial runs: {total}")
    print("=" * 70)

    all_results = []
    config_idx = 0
    total_configs = (len(grid['datasets']) * len(grid['partition_methods']) *
                     len(grid['client_counts']) * len(grid['methods']))

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
        print(f"  Nodes={data.num_nodes}, Edges={data.edge_index.size(1)}, "
              f"Features={num_features}, Classes={num_classes}, H={homophily:.3f}")

        model_template = GCN2Layer(num_features, num_classes).to(device)

        for partition_method in grid['partition_methods']:
            for num_clients in grid['client_counts']:
                # Partition
                partition_result = partition_graph(
                    data, num_clients, method=partition_method, seed=args.seed,
                )
                print(f"\n  Partition: {partition_method}, K={num_clients}, "
                      f"cross_ratio={partition_result.cross_edge_ratio:.3f}")

                # Build subgraphs
                client_subgraphs = {}
                for cid in range(num_clients):
                    nodes = (partition_result.partition_map == cid).nonzero(as_tuple=True)[0]
                    client_subgraphs[cid] = build_client_subgraph(
                        data, nodes, data.edge_index, cid
                    )

                for method in grid['methods']:
                    config_idx += 1
                    print(f"\n  [{config_idx}/{total_configs}] {method}")

                    method_results = []
                    t_start = time.time()

                    for trial in range(args.trials):
                        res = run_trial(
                            data, partition_result, client_subgraphs,
                            model_template, device, method,
                            trial_idx=trial, grid=grid, seed=args.seed,
                        )
                        if res is None:
                            continue

                        res.update({
                            'dataset': dataset_name,
                            'partition': partition_method,
                            'num_clients': num_clients,
                            'method': method,
                            'trial': trial,
                            'homophily': homophily,
                        })
                        method_results.append(res)
                        all_results.append(res)

                    elapsed = time.time() - t_start
                    n = len(method_results)

                    if n == 0:
                        print(f"    No valid trials!")
                        continue

                    # Per-method summary
                    gl2 = np.mean([r['global_l2_auc'] for r in method_results])
                    gc = np.mean([r['global_conf_auc'] for r in method_results])
                    gg = np.mean([r['global_gap'] for r in method_results])
                    ll2 = np.mean([r['local_l2_auc'] for r in method_results])
                    cl2 = np.mean([r['mean_cross_l2_auc'] for r in method_results])
                    mx = np.mean([r['max_cross_l2_auc'] for r in method_results])

                    print(f"    {n} trials, {elapsed:.1f}s | "
                          f"G-L2={gl2:.3f} G-Conf={gc:.3f} Gap={gg:+.3f} | "
                          f"L-L2={ll2:.3f} | X-L2={cl2:.3f} X-max={mx:.3f}")

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
        output_path = os.path.join(output_dir, f'main_results_{timestamp}.csv')

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # ================================================================
    # Summary tables
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Mean AUC by Method × Dataset")
    print("=" * 70)

    # Table 1: Global L2 AUC
    print("\n[Table 1] Global L2 AUC (core privacy leakage metric)")
    pivot_gl2 = df.pivot_table(
        values='global_l2_auc', index='method', columns='dataset', aggfunc='mean'
    )
    print(pivot_gl2.round(4).to_string())

    # Table 2: Global Conf AUC
    print("\n[Table 2] Global Conf AUC (confidence illusion)")
    pivot_gc = df.pivot_table(
        values='global_conf_auc', index='method', columns='dataset', aggfunc='mean'
    )
    print(pivot_gc.round(4).to_string())

    # Table 3: Gap (L2 - Conf)
    print("\n[Table 3] Gap = L2 - Conf (positive = illusion detected)")
    pivot_gap = df.pivot_table(
        values='global_gap', index='method', columns='dataset', aggfunc='mean'
    )
    print(pivot_gap.round(4).to_string())

    # Table 4: Local L2 AUC
    print("\n[Table 4] Local L2 AUC (unlearning client's local model)")
    pivot_ll2 = df.pivot_table(
        values='local_l2_auc', index='method', columns='dataset', aggfunc='mean'
    )
    print(pivot_ll2.round(4).to_string())

    # Table 5: Cross-client L2 AUC
    print("\n[Table 5] Cross-Client L2 AUC (leakage to other clients)")
    pivot_cl2 = df.pivot_table(
        values='mean_cross_l2_auc', index='method', columns='dataset', aggfunc='mean'
    )
    print(pivot_cl2.round(4).to_string())

    # Table 6: Multi-level comparison (aggregated across datasets)
    print("\n[Table 6] Multi-Level Comparison (all datasets)")
    print(f"{'Method':<20} {'Global L2':>10} {'Local L2':>10} {'Cross L2':>10} "
          f"{'Global Conf':>12} {'Gap':>8}")
    print("─" * 72)
    for method in df['method'].unique():
        mdf = df[df['method'] == method]
        print(f"{method:<20} "
              f"{mdf['global_l2_auc'].mean():>10.4f} "
              f"{mdf['local_l2_auc'].mean():>10.4f} "
              f"{mdf['mean_cross_l2_auc'].mean():>10.4f} "
              f"{mdf['global_conf_auc'].mean():>12.4f} "
              f"{mdf['global_gap'].mean():>+8.4f}")

    # RQ verification
    print("\n" + "=" * 70)
    print("Research Question Verification")
    print("=" * 70)

    # RQ1: Confidence Illusion
    approx_methods = df[df['method'] != 'FedRetrain']
    if len(approx_methods) > 0:
        avg_gap = approx_methods['global_gap'].mean()
        print(f"\nRQ1 (Confidence Illusion): Avg Gap (approx methods) = {avg_gap:+.4f}")
        print(f"  → {'CONFIRMED' if avg_gap > 0.05 else 'WEAK'}: "
              f"L2 detects leakage that confidence misses")

    # RQ3: Level ordering
    for method in df['method'].unique():
        mdf = df[df['method'] == method]
        g = mdf['global_l2_auc'].mean()
        l = mdf['local_l2_auc'].mean()
        c = mdf['mean_cross_l2_auc'].mean()
        ordering = f"Local({l:.3f}) {'>' if l > g else '<'} " \
                   f"Global({g:.3f}) {'>' if g > c else '<'} Cross({c:.3f})"
        print(f"\nRQ3 ({method}): {ordering}")


if __name__ == '__main__':
    main()
