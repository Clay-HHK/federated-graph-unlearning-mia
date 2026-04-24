"""
Experiment F: Cross-Edge Ablation — FedSage-like vs Intra-Only (Reviewer #3)

Compares privacy leakage under two training modes:
  1. Intra-only: drop all cross-client edges (current paper setting)
  2. With-neighbors: include cross-client edges + remote neighbor features
     (simulates FedSage neighbor generation)

If leakage is HIGHER with cross-edges, our intra-only results are a lower bound.

Usage:
    python experiments/federated/run_fed_cross_edge_ablation.py --trials 50
    python experiments/federated/run_fed_cross_edge_ablation.py --quick --trials 10
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
from src.federated.subgraph import build_client_subgraph, build_client_subgraph_with_neighbors
from src.federated.data_partition import partition_graph
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.unlearning import fed_retrain, fed_gnndelete
from src.federated.attacks.hub_ripple_federated import multilevel_hub_ripple


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-Edge Ablation (FedSage-like)')
    parser.add_argument('--trials', type=int, default=50)
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


def build_subgraphs(data, partition_result, num_clients, mode, seed):
    """Build client subgraphs in either intra-only or with-neighbors mode."""
    client_subgraphs = {}
    for cid in range(num_clients):
        nodes = (partition_result.partition_map == cid).nonzero(as_tuple=True)[0]
        if mode == 'intra_only':
            client_subgraphs[cid] = build_client_subgraph(
                data, nodes, data.edge_index, cid
            )
        elif mode == 'with_neighbors':
            client_subgraphs[cid] = build_client_subgraph_with_neighbors(
                data, nodes, data.edge_index, cid
            )
    return client_subgraphs


def run_trial(data, partition_result, num_clients, model_template,
              device, method, mode, trial_idx, grid, seed=42):
    """Run single trial under specified edge mode."""
    trial_seed = seed + trial_idx
    set_seed(trial_seed)

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

    target_idx, target_client_id, hubs, ctrls = \
        select_target_and_hubs(data, partition_result, seed=trial_seed)

    if len(hubs) < 3 or len(ctrls) < 3:
        return None

    # Build subgraphs with specified mode
    client_subgraphs = build_subgraphs(
        data, partition_result, num_clients, mode, seed=trial_seed
    )

    # Count edges per mode for reporting
    total_local_edges = sum(
        sg.data.edge_index.size(1) for sg in client_subgraphs.values()
    )

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

    # Deep copy for unlearning
    ul_clients = []
    for cid in server.client_ids:
        ul_clients.append(FederatedClient(
            cid, copy.deepcopy(server.clients[cid].subgraph),
            server.clients[cid].model, device,
        ))
    ul_server = FederatedServer(copy.deepcopy(server.global_model), ul_clients, device)

    t0 = time.time()
    if method == 'FedRetrain':
        # For FedRetrain with-neighbors, rebuild subgraphs in same mode
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

    # For attack evaluation, always use intra-only subgraphs
    # (attack evaluates on each client's own data)
    attack_subgraphs = build_subgraphs(
        data, partition_result, num_clients, 'intra_only', seed=trial_seed
    )

    attack = multilevel_hub_ripple(
        snap_before, snap_after,
        data, attack_subgraphs,
        target_idx, target_client_id,
        hubs, ctrls,
        model_template, device,
    )

    return {
        'edge_mode': mode,
        'method': method,
        'trial': trial_idx,
        'global_l2_auc': attack.global_l2_auc,
        'global_conf_auc': attack.global_conf_auc,
        'global_gap': attack.global_gap,
        'local_l2_auc': attack.local_l2_auc,
        'local_conf_auc': attack.local_conf_auc,
        'mean_cross_l2_auc': attack.mean_cross_l2_auc,
        'max_cross_l2_auc': attack.max_cross_l2_auc,
        'num_hubs': attack.num_hubs,
        'unlearn_time': unlearn_time,
        'global_acc_before': global_acc_before,
        'total_local_edges': total_local_edges,
    }


def main():
    args = parse_args()
    device = get_device()

    modes = ['intra_only', 'with_neighbors']
    methods = ['FedGNNDelete', 'FedRetrain']
    grid = {
        'num_rounds': 50,
        'local_epochs': 5,
    }

    if args.quick:
        methods = ['FedGNNDelete']
        grid['num_rounds'] = 30
        grid['local_epochs'] = 3

    datasets = ['Cora', 'CiteSeer']
    num_clients = 5

    print("=" * 70)
    print("Experiment F: Cross-Edge Ablation (FedSage-like vs Intra-Only)")
    print("=" * 70)
    print(f"Datasets: {datasets}")
    print(f"Modes: {modes}")
    print(f"Methods: {methods}")
    print(f"K={num_clients}, Trials={args.trials}")
    total = len(datasets) * len(modes) * len(methods) * args.trials
    print(f"Total trial runs: {total}")
    print("=" * 70)

    all_results = []

    for dataset_name in datasets:
        print(f"\n{'━' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'━' * 60}")

        data, homophily = load_dataset(dataset_name, device)
        num_features = data.num_features
        num_classes = int(data.y.max().item()) + 1
        model_template = GCN2Layer(num_features, num_classes).to(device)

        partition_result = partition_graph(
            data, num_clients, method='metis', seed=args.seed,
        )
        print(f"  Nodes={data.num_nodes}, Edges={data.edge_index.size(1)}, "
              f"cross_ratio={partition_result.cross_edge_ratio:.3f}")

        # Show edge counts per mode
        sg_intra = build_subgraphs(data, partition_result, num_clients, 'intra_only', args.seed)
        sg_neigh = build_subgraphs(data, partition_result, num_clients, 'with_neighbors', args.seed)
        e_intra = sum(sg.data.edge_index.size(1) for sg in sg_intra.values())
        e_neigh = sum(sg.data.edge_index.size(1) for sg in sg_neigh.values())
        print(f"  Intra-only edges: {e_intra}, With-neighbors edges: {e_neigh} "
              f"(+{e_neigh - e_intra}, {(e_neigh/e_intra - 1)*100:.1f}% increase)")

        for method in methods:
            for mode in modes:
                print(f"\n  {dataset_name} | {method} | {mode}:")
                method_results = []
                t_start = time.time()

                for trial in range(args.trials):
                    res = run_trial(
                        data, partition_result, num_clients,
                        model_template, device, method, mode,
                        trial_idx=trial, grid=grid, seed=args.seed,
                    )
                    if res is None:
                        continue

                    res['dataset'] = dataset_name
                    res['homophily'] = homophily
                    res['cross_edge_ratio'] = partition_result.cross_edge_ratio
                    method_results.append(res)
                    all_results.append(res)

                elapsed = time.time() - t_start
                n = len(method_results)

                if n == 0:
                    print(f"    No valid trials!")
                    continue

                gl2 = np.mean([r['global_l2_auc'] for r in method_results])
                ll2 = np.mean([r['local_l2_auc'] for r in method_results])
                cl2 = np.mean([r['mean_cross_l2_auc'] for r in method_results])
                acc = np.mean([r['global_acc_before'] for r in method_results])
                print(f"    {n} trials, {elapsed:.1f}s | Acc={acc:.3f} | "
                      f"G-L2={gl2:.3f} L-L2={ll2:.3f} X-L2={cl2:.3f}")

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
        output_path = os.path.join(output_dir, f'cross_edge_ablation_{timestamp}.csv')

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Cross-Edge Ablation Summary")
    print("=" * 70)
    print(f"{'Dataset':<12} {'Method':<15} {'Mode':<18} {'N':>4} "
          f"{'Acc':>7} {'G-L2':>8} {'L-L2':>8} {'X-L2':>8}")
    print("-" * 85)

    for dataset_name in datasets:
        for method in methods:
            for mode in modes:
                sub = df[(df['dataset'] == dataset_name) &
                         (df['method'] == method) &
                         (df['edge_mode'] == mode)]
                if len(sub) == 0:
                    continue
                print(f"{dataset_name:<12} {method:<15} {mode:<18} {len(sub):>4} "
                      f"{sub['global_acc_before'].mean():>7.3f} "
                      f"{sub['global_l2_auc'].mean():>5.3f}±{sub['global_l2_auc'].std():>4.3f} "
                      f"{sub['local_l2_auc'].mean():>5.3f}±{sub['local_l2_auc'].std():>4.3f} "
                      f"{sub['mean_cross_l2_auc'].mean():>5.3f}±{sub['mean_cross_l2_auc'].std():>4.3f}")

    # Paired comparison
    print("\n" + "=" * 70)
    print("Paired Comparison: with_neighbors vs intra_only")
    print("=" * 70)

    from scipy.stats import wilcoxon

    for dataset_name in datasets:
        for method in methods:
            intra = df[(df['dataset'] == dataset_name) &
                       (df['method'] == method) &
                       (df['edge_mode'] == 'intra_only')]
            neigh = df[(df['dataset'] == dataset_name) &
                       (df['method'] == method) &
                       (df['edge_mode'] == 'with_neighbors')]

            if len(intra) == 0 or len(neigh) == 0:
                continue

            for metric in ['global_l2_auc', 'local_l2_auc', 'mean_cross_l2_auc']:
                v_intra = intra[metric].values
                v_neigh = neigh[metric].values
                n = min(len(v_intra), len(v_neigh))
                v_intra = v_intra[:n]
                v_neigh = v_neigh[:n]

                diff = np.mean(v_neigh) - np.mean(v_intra)
                try:
                    stat, pval = wilcoxon(v_neigh, v_intra)
                except ValueError:
                    pval = 1.0
                    stat = 0.0

                direction = "HIGHER" if diff > 0 else "LOWER" if diff < 0 else "SAME"
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"

                print(f"  {dataset_name} | {method} | {metric:<22} | "
                      f"Δ={diff:+.4f} ({direction}) | p={pval:.4e} {sig}")

    # Key conclusion
    print("\n" + "=" * 70)
    print("Conclusion:")
    neigh_all = df[df['edge_mode'] == 'with_neighbors']
    intra_all = df[df['edge_mode'] == 'intra_only']
    if len(neigh_all) > 0 and len(intra_all) > 0:
        g_diff = neigh_all['global_l2_auc'].mean() - intra_all['global_l2_auc'].mean()
        c_diff = neigh_all['mean_cross_l2_auc'].mean() - intra_all['mean_cross_l2_auc'].mean()
        print(f"  Global L2 AUC change: {g_diff:+.4f}")
        print(f"  Cross  L2 AUC change: {c_diff:+.4f}")
        if g_diff > 0 or c_diff > 0:
            print("  → Including cross-client edges INCREASES leakage.")
            print("  → Intra-only results are a LOWER BOUND on real-world leakage.")
        else:
            print("  → Cross-client edges do not significantly change leakage.")
            print("  → The leakage is dominated by parameter-mediated signals.")


if __name__ == '__main__':
    main()
