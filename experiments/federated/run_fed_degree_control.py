"""
Experiment B: Degree-Stratified Control Node Analysis (Reviewer #4)

Compares uniform vs degree-matched control node sampling to verify
that the observed AUC is not an artifact of degree disparity.

Uses shared training phase: same trained model, two attack strategies.

Usage:
    python experiments/federated/run_fed_degree_control.py --trials 50
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
    get_neighbors, select_control_node,
    select_control_node_degree_matched, get_node_degrees,
)
from src.models.gcn import GCN2Layer
from src.models.training import get_embeddings
from src.federated.subgraph import build_client_subgraph
from src.federated.data_partition import partition_graph
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.unlearning import fed_gnndelete
from src.federated.attacks.hub_ripple_federated import multilevel_hub_ripple


def parse_args():
    parser = argparse.ArgumentParser(description='Degree-Stratified Control Analysis')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


def run_trial(data, partition_result, client_subgraphs, model_template,
              device, trial_idx, grid, seed=42):
    """Run single trial with both control strategies.

    Returns two result dicts (uniform, degree_matched) or (None, None).
    """
    trial_seed = seed + trial_idx
    set_seed(trial_seed)

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1
    degrees = get_node_degrees(data.edge_index, data.num_nodes)

    # Select target
    sorted_nodes = torch.argsort(degrees, descending=True)
    target_idx = None
    for candidate in sorted_nodes[:100].tolist():
        neighbors = get_neighbors(data.edge_index, candidate)
        if len(neighbors) >= 5:
            target_idx = candidate
            break
    if target_idx is None:
        return None, None

    target_client_id = partition_result.partition_map[target_idx].item()
    neighbors = get_neighbors(data.edge_index, target_idx)
    hub_indices = neighbors.tolist()

    # Generate both control sets
    uniform_controls = []
    matched_controls = []
    for hub_idx in hub_indices:
        try:
            ctrl_u = select_control_node(
                data.edge_index, data.num_nodes,
                hub_idx, target_idx, seed=trial_seed + hub_idx
            )
            ctrl_m = select_control_node_degree_matched(
                data.edge_index, data.num_nodes,
                hub_idx, target_idx, seed=trial_seed + hub_idx,
                degree_tolerance=3,
            )
            uniform_controls.append(ctrl_u)
            matched_controls.append(ctrl_m)
        except ValueError:
            continue

    min_len = min(len(hub_indices), len(uniform_controls), len(matched_controls))
    if min_len < 3:
        return None, None

    hubs = hub_indices[:min_len]
    u_ctrls = uniform_controls[:min_len]
    m_ctrls = matched_controls[:min_len]

    # Compute degree statistics
    hub_degrees = [degrees[h].item() for h in hubs]
    u_ctrl_degrees = [degrees[c].item() for c in u_ctrls]
    m_ctrl_degrees = [degrees[c].item() for c in m_ctrls]

    # Train model (shared)
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

    # Unlearn (shared)
    ul_clients = []
    for cid in server.client_ids:
        ul_clients.append(FederatedClient(
            cid, copy.deepcopy(server.clients[cid].subgraph),
            server.clients[cid].model, device,
        ))
    ul_server = FederatedServer(copy.deepcopy(server.global_model), ul_clients, device)

    result = fed_gnndelete(
        ul_server, target_idx, target_client_id,
        num_features, num_classes,
        reaggregate_rounds=5, local_epochs=grid['local_epochs'],
    )
    ul_server = result.server
    ul_server.snapshot_state('after')

    snap_before = server.get_snapshot('before')
    snap_after = ul_server.get_snapshot('after')

    # Attack with uniform controls
    attack_u = multilevel_hub_ripple(
        snap_before, snap_after,
        data, client_subgraphs,
        target_idx, target_client_id,
        hubs, u_ctrls,
        model_template, device,
    )

    # Attack with degree-matched controls
    attack_m = multilevel_hub_ripple(
        snap_before, snap_after,
        data, client_subgraphs,
        target_idx, target_client_id,
        hubs, m_ctrls,
        model_template, device,
    )

    base_info = {
        'trial': trial_idx,
        'target_idx': target_idx,
        'num_hubs': len(hubs),
        'hub_mean_degree': float(np.mean(hub_degrees)),
        'hub_median_degree': float(np.median(hub_degrees)),
        'global_acc_before': global_acc_before,
    }

    res_uniform = {
        **base_info,
        'control_strategy': 'uniform',
        'global_l2_auc': attack_u.global_l2_auc,
        'global_conf_auc': attack_u.global_conf_auc,
        'local_l2_auc': attack_u.local_l2_auc,
        'mean_cross_l2_auc': attack_u.mean_cross_l2_auc,
        'control_mean_degree': float(np.mean(u_ctrl_degrees)),
        'control_median_degree': float(np.median(u_ctrl_degrees)),
        'degree_gap': float(np.mean(hub_degrees) - np.mean(u_ctrl_degrees)),
    }

    res_matched = {
        **base_info,
        'control_strategy': 'degree_matched',
        'global_l2_auc': attack_m.global_l2_auc,
        'global_conf_auc': attack_m.global_conf_auc,
        'local_l2_auc': attack_m.local_l2_auc,
        'mean_cross_l2_auc': attack_m.mean_cross_l2_auc,
        'control_mean_degree': float(np.mean(m_ctrl_degrees)),
        'control_median_degree': float(np.median(m_ctrl_degrees)),
        'degree_gap': float(np.mean(hub_degrees) - np.mean(m_ctrl_degrees)),
    }

    return res_uniform, res_matched


def main():
    args = parse_args()
    device = get_device()

    grid = {
        'num_rounds': 50,
        'local_epochs': 5,
    }
    dataset_name = 'Cora'
    num_clients = 5

    print("=" * 70)
    print("Experiment B: Degree-Stratified Control Node Analysis")
    print("=" * 70)
    print(f"Dataset: {dataset_name}, K={num_clients}, Method: FedGNNDelete")
    print(f"Trials: {args.trials}")
    print("=" * 70)

    data, homophily = load_dataset(dataset_name, device)
    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1
    model_template = GCN2Layer(num_features, num_classes).to(device)

    partition_result = partition_graph(
        data, num_clients, method='metis', seed=args.seed,
    )

    client_subgraphs = {}
    for cid in range(num_clients):
        nodes = (partition_result.partition_map == cid).nonzero(as_tuple=True)[0]
        client_subgraphs[cid] = build_client_subgraph(
            data, nodes, data.edge_index, cid
        )

    all_results = []

    for trial in range(args.trials):
        res_u, res_m = run_trial(
            data, partition_result, client_subgraphs,
            model_template, device,
            trial_idx=trial, grid=grid, seed=args.seed,
        )
        if res_u is not None:
            all_results.append(res_u)
            all_results.append(res_m)

        if (trial + 1) % 10 == 0:
            n_valid = len([r for r in all_results if r['control_strategy'] == 'uniform'])
            print(f"  Trial {trial+1}/{args.trials}: {n_valid} valid pairs")

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
        output_path = os.path.join(output_dir, f'degree_matched_{timestamp}.csv')

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Degree-Stratified Control Analysis Summary")
    print("=" * 70)

    for strategy in ['uniform', 'degree_matched']:
        sub = df[df['control_strategy'] == strategy]
        print(f"\n{strategy.upper()} (n={len(sub)}):")
        print(f"  Global L2 AUC:  {sub['global_l2_auc'].mean():.4f} ± {sub['global_l2_auc'].std():.4f}")
        print(f"  Local L2 AUC:   {sub['local_l2_auc'].mean():.4f} ± {sub['local_l2_auc'].std():.4f}")
        print(f"  Cross L2 AUC:   {sub['mean_cross_l2_auc'].mean():.4f} ± {sub['mean_cross_l2_auc'].std():.4f}")
        print(f"  Hub mean deg:   {sub['hub_mean_degree'].mean():.1f}")
        print(f"  Ctrl mean deg:  {sub['control_mean_degree'].mean():.1f}")
        print(f"  Degree gap:     {sub['degree_gap'].mean():.1f}")

    # Paired Wilcoxon test
    from scipy.stats import wilcoxon
    u_auc = df[df['control_strategy'] == 'uniform']['global_l2_auc'].values
    m_auc = df[df['control_strategy'] == 'degree_matched']['global_l2_auc'].values

    if len(u_auc) == len(m_auc) and len(u_auc) > 0:
        stat, pval = wilcoxon(u_auc, m_auc)
        diff = np.mean(u_auc) - np.mean(m_auc)
        print(f"\nPaired Wilcoxon signed-rank test:")
        print(f"  Mean difference (uniform - matched): {diff:+.4f}")
        print(f"  Statistic: {stat:.2f}")
        print(f"  p-value: {pval:.4e}")
        print(f"  Significant (p<0.05): {'Yes' if pval < 0.05 else 'No'}")

        if abs(diff) < 0.03:
            print("  Conclusion: Degree matching does NOT substantially change AUC.")
            print("  → Uniform sampling is conservative and does not overestimate leakage.")


if __name__ == '__main__':
    main()
