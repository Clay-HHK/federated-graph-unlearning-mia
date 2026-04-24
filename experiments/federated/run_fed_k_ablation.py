"""
Experiment A: K-Value Ablation (Reviewer #6)

Tests the 1/K dilution hypothesis by varying the number of federated
clients K = {3, 5, 10, 20} on Cora with FedRetrain and FedGNNDelete.

Usage:
    python experiments/federated/run_fed_k_ablation.py --trials 50
    python experiments/federated/run_fed_k_ablation.py --quick --trials 10
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
from src.utils.graph_utils import get_neighbors, select_control_node, get_node_degrees
from src.models.gcn import GCN2Layer
from src.federated.subgraph import build_client_subgraph
from src.federated.data_partition import partition_graph
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.unlearning import fed_retrain, fed_gnndelete
from src.federated.attacks.hub_ripple_federated import multilevel_hub_ripple


def parse_args():
    parser = argparse.ArgumentParser(description='K-Value Ablation Experiment')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


def select_target_and_hubs(data, partition_result, seed=42, min_hubs=3):
    """Select target node and hub/control pairs.

    Uses min_hubs=3 (lower than main experiment's 5) to accommodate
    small per-client subgraphs when K is large.
    """
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
    """Run single trial for a given K configuration."""
    trial_seed = seed + trial_idx
    set_seed(trial_seed)

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

    target_idx, target_client_id, hubs, ctrls = \
        select_target_and_hubs(data, partition_result, seed=trial_seed, min_hubs=3)

    if len(hubs) < 3 or len(ctrls) < 3:
        return None

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

    global_acc_before = server.evaluate_global(data)
    server.snapshot_state('before')

    ul_clients = []
    for cid in server.client_ids:
        ul_clients.append(FederatedClient(
            cid, copy.deepcopy(server.clients[cid].subgraph),
            server.clients[cid].model, device,
        ))
    ul_server = FederatedServer(copy.deepcopy(server.global_model), ul_clients, device)

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
    else:
        return None
    unlearn_time = time.time() - t0

    ul_server = result.server
    ul_server.snapshot_state('after')

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
        'mean_cross_l2_auc': attack.mean_cross_l2_auc,
        'max_cross_l2_auc': attack.max_cross_l2_auc,
        'num_hubs': attack.num_hubs,
        'unlearn_time': unlearn_time,
        'global_acc_before': global_acc_before,
    }


def main():
    args = parse_args()
    device = get_device()

    client_counts = [3, 5, 10, 20]
    methods = ['FedRetrain', 'FedGNNDelete']
    grid = {
        'num_rounds': 50,
        'local_epochs': 5,
        'epochs': 200,
    }
    if args.quick:
        client_counts = [3, 5, 10]
        grid['num_rounds'] = 30
        grid['local_epochs'] = 3

    dataset_name = 'Cora'

    print("=" * 70)
    print("Experiment A: K-Value Ablation")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"K values: {client_counts}")
    print(f"Methods: {methods}")
    print(f"Trials: {args.trials}")
    total = len(client_counts) * len(methods) * args.trials
    print(f"Total trial runs: {total}")
    print("=" * 70)

    data, homophily = load_dataset(dataset_name, device)
    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1
    print(f"Nodes={data.num_nodes}, Edges={data.edge_index.size(1)}, "
          f"H={homophily:.3f}")

    model_template = GCN2Layer(num_features, num_classes).to(device)
    all_results = []

    for num_clients in client_counts:
        partition_result = partition_graph(
            data, num_clients, method='metis', seed=args.seed,
        )
        print(f"\nK={num_clients}, cross_ratio={partition_result.cross_edge_ratio:.3f}")

        client_subgraphs = {}
        for cid in range(num_clients):
            nodes = (partition_result.partition_map == cid).nonzero(as_tuple=True)[0]
            client_subgraphs[cid] = build_client_subgraph(
                data, nodes, data.edge_index, cid
            )
            print(f"  Client {cid}: {len(nodes)} nodes")

        for method in methods:
            print(f"\n  K={num_clients}, {method}:")
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
                    'partition': 'metis',
                    'num_clients': num_clients,
                    'method': method,
                    'trial': trial,
                    'homophily': homophily,
                    'cross_edge_ratio': partition_result.cross_edge_ratio,
                })
                method_results.append(res)
                all_results.append(res)

            elapsed = time.time() - t_start
            n = len(method_results)

            if n == 0:
                print(f"    No valid trials!")
                continue

            gl2 = np.mean([r['global_l2_auc'] for r in method_results])
            gc = np.mean([r['global_conf_auc'] for r in method_results])
            cl2 = np.mean([r['mean_cross_l2_auc'] for r in method_results])
            print(f"    {n} trials, {elapsed:.1f}s | "
                  f"G-L2={gl2:.3f} G-Conf={gc:.3f} X-L2={cl2:.3f}")

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
        output_path = os.path.join(output_dir, f'k_ablation_{timestamp}.csv')

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("K-Ablation Summary: Mean ± Std by K")
    print("=" * 70)
    print(f"{'K':>4} {'Method':<15} {'N':>4} "
          f"{'Global L2':>12} {'Global Conf':>12} {'Gap':>10} "
          f"{'Local L2':>12} {'Cross L2':>12}")
    print("-" * 90)

    for k in client_counts:
        for method in methods:
            sub = df[(df['num_clients'] == k) & (df['method'] == method)]
            if len(sub) == 0:
                continue
            print(f"{k:>4} {method:<15} {len(sub):>4} "
                  f"{sub['global_l2_auc'].mean():>6.3f}±{sub['global_l2_auc'].std():>4.3f} "
                  f"{sub['global_conf_auc'].mean():>6.3f}±{sub['global_conf_auc'].std():>4.3f} "
                  f"{sub['global_gap'].mean():>+5.3f}±{sub['global_gap'].std():>4.3f} "
                  f"{sub['local_l2_auc'].mean():>6.3f}±{sub['local_l2_auc'].std():>4.3f} "
                  f"{sub['mean_cross_l2_auc'].mean():>6.3f}±{sub['mean_cross_l2_auc'].std():>4.3f}")

    # 1/K dilution check
    print("\n1/K Dilution Analysis (FedGNNDelete):")
    gnndelete = df[df['method'] == 'FedGNNDelete']
    for k in client_counts:
        sub = gnndelete[gnndelete['num_clients'] == k]
        if len(sub) > 0:
            print(f"  K={k:>2}: Global L2 AUC = {sub['global_l2_auc'].mean():.4f}, "
                  f"1/K = {1/k:.4f}")


if __name__ == '__main__':
    main()
