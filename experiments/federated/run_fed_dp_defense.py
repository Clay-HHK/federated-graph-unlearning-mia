"""
Experiment E: DP-SGD Defense Evaluation (Reviewer #8)

Evaluates whether differential privacy (gradient clipping + Gaussian noise)
mitigates the Hub-Ripple attack. Tests privacy-utility tradeoff across
epsilon = {1, 5, 10, inf}.

Usage:
    python experiments/federated/run_fed_dp_defense.py --trials 50
    python experiments/federated/run_fed_dp_defense.py --quick --trials 10
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
from src.models.training import evaluate_model
from src.federated.subgraph import build_client_subgraph
from src.federated.data_partition import partition_graph
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.unlearning import fed_gnndelete
from src.federated.attacks.hub_ripple_federated import multilevel_hub_ripple
from src.federated.dp_sgd import DPConfig


def parse_args():
    parser = argparse.ArgumentParser(description='DP-SGD Defense Evaluation')
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


def train_federated_with_dp(server, dp_config, num_rounds, local_epochs, lr=0.01):
    """Train federated model with DP-SGD on all clients.

    Args:
        server: FederatedServer instance
        dp_config: DPConfig or None (None means no DP, standard training)
        num_rounds: Number of federated rounds
        local_epochs: Local epochs per round
        lr: Learning rate
    """
    for round_idx in range(num_rounds):
        # Broadcast
        server.broadcast_global_model()

        # Local training
        client_states = []
        client_weights = []
        for cid in server.client_ids:
            client = server.clients[cid]

            if dp_config is not None:
                client.train_local_dp(
                    dp_config, epochs=local_epochs, lr=lr, weight_decay=5e-4
                )
            else:
                client.train_local(
                    epochs=local_epochs, lr=lr, weight_decay=5e-4
                )

            client_states.append(client.upload_model())
            client_weights.append(float(client.num_nodes))

        # Aggregate
        avg_state = server.fedavg_aggregate(client_states, client_weights)
        server.global_model.load_state_dict(avg_state)

    # Final broadcast
    server.broadcast_global_model()


def run_trial(data, partition_result, client_subgraphs, model_template,
              device, epsilon, trial_idx, grid, seed=42):
    """Run single trial with specified DP epsilon."""
    trial_seed = seed + trial_idx
    set_seed(trial_seed)

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

    target_idx, target_client_id, hubs, ctrls = \
        select_target_and_hubs(data, partition_result, seed=trial_seed)

    if len(hubs) < 3 or len(ctrls) < 3:
        return None

    # Configure DP
    if epsilon == float('inf'):
        dp_config = None
    else:
        dp_config = DPConfig(
            epsilon=epsilon,
            delta=1e-5,
            max_grad_norm=1.0,
        )

    global_model = GCN2Layer(num_features, num_classes).to(device)
    set_seed(trial_seed)
    global_model.reset_parameters()

    clients = []
    for cid, subgraph in client_subgraphs.items():
        clients.append(FederatedClient(cid, copy.deepcopy(subgraph), global_model, device))

    server = FederatedServer(global_model, clients, device)

    # Train with DP
    train_federated_with_dp(
        server, dp_config,
        num_rounds=grid['num_rounds'],
        local_epochs=grid['local_epochs'],
    )

    global_acc_before = server.evaluate_global(data)
    server.snapshot_state('before')

    # Unlearn (always without DP for fair comparison)
    ul_clients = []
    for cid in server.client_ids:
        ul_clients.append(FederatedClient(
            cid, copy.deepcopy(server.clients[cid].subgraph),
            server.clients[cid].model, device,
        ))
    ul_server = FederatedServer(copy.deepcopy(server.global_model), ul_clients, device)

    t0 = time.time()
    result = fed_gnndelete(
        ul_server, target_idx, target_client_id,
        num_features, num_classes,
        reaggregate_rounds=5, local_epochs=grid['local_epochs'],
    )
    unlearn_time = time.time() - t0

    ul_server = result.server
    ul_server.snapshot_state('after')

    global_acc_after = ul_server.evaluate_global(data)

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
        'epsilon': epsilon,
        'delta': 1e-5 if epsilon != float('inf') else None,
        'max_grad_norm': 1.0 if epsilon != float('inf') else None,
        'trial': trial_idx,
        'global_l2_auc': attack.global_l2_auc,
        'global_conf_auc': attack.global_conf_auc,
        'global_gap': attack.global_gap,
        'local_l2_auc': attack.local_l2_auc,
        'mean_cross_l2_auc': attack.mean_cross_l2_auc,
        'max_cross_l2_auc': attack.max_cross_l2_auc,
        'num_hubs': attack.num_hubs,
        'unlearn_time': unlearn_time,
        'global_acc_before': global_acc_before,
        'global_acc_after': global_acc_after,
    }


def main():
    args = parse_args()
    device = get_device()

    epsilons = [1.0, 5.0, 10.0, float('inf')]
    grid = {
        'num_rounds': 50,
        'local_epochs': 5,
    }
    if args.quick:
        epsilons = [1.0, 10.0, float('inf')]
        grid['num_rounds'] = 30
        grid['local_epochs'] = 3

    dataset_name = 'Cora'
    num_clients = 5

    print("=" * 70)
    print("Experiment E: DP-SGD Defense Evaluation")
    print("=" * 70)
    print(f"Dataset: {dataset_name}, K={num_clients}, Method: FedGNNDelete")
    print(f"Epsilons: {epsilons}")
    print(f"Trials: {args.trials}")
    total = len(epsilons) * args.trials
    print(f"Total trial runs: {total}")
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

    for epsilon in epsilons:
        eps_str = f"ε={epsilon}" if epsilon != float('inf') else "ε=∞ (no DP)"
        print(f"\n{'━' * 60}")
        print(f"Testing {eps_str}")
        if epsilon != float('inf'):
            dp = DPConfig(epsilon=epsilon)
            print(f"  noise_multiplier σ = {dp.noise_multiplier:.4f}")
        print(f"{'━' * 60}")

        method_results = []
        t_start = time.time()

        for trial in range(args.trials):
            res = run_trial(
                data, partition_result, client_subgraphs,
                model_template, device, epsilon,
                trial_idx=trial, grid=grid, seed=args.seed,
            )
            if res is None:
                continue

            res['dataset'] = dataset_name
            method_results.append(res)
            all_results.append(res)

        elapsed = time.time() - t_start
        n = len(method_results)

        if n == 0:
            print(f"  No valid trials!")
            continue

        gl2 = np.mean([r['global_l2_auc'] for r in method_results])
        gc = np.mean([r['global_conf_auc'] for r in method_results])
        acc_b = np.mean([r['global_acc_before'] for r in method_results])
        acc_a = np.mean([r['global_acc_after'] for r in method_results])
        cl2 = np.mean([r['mean_cross_l2_auc'] for r in method_results])

        print(f"  {n} trials, {elapsed:.1f}s")
        print(f"  Acc before: {acc_b:.4f}, Acc after: {acc_a:.4f}")
        print(f"  Global L2 AUC: {gl2:.4f}, Conf AUC: {gc:.4f}")
        print(f"  Cross L2 AUC: {cl2:.4f}")

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
        output_path = os.path.join(output_dir, f'dp_defense_{timestamp}.csv')

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("DP-SGD Defense Summary: Privacy-Utility Tradeoff")
    print("=" * 70)
    print(f"{'Epsilon':>10} {'N':>4} {'Acc Before':>12} {'Acc After':>12} "
          f"{'Global L2':>12} {'Global Conf':>12} {'Cross L2':>12}")
    print("-" * 80)

    for eps in epsilons:
        sub = df[df['epsilon'] == eps]
        if len(sub) == 0:
            continue
        eps_str = f"{eps:.1f}" if eps != float('inf') else "inf"
        print(f"{eps_str:>10} {len(sub):>4} "
              f"{sub['global_acc_before'].mean():>6.4f}±{sub['global_acc_before'].std():>4.4f} "
              f"{sub['global_acc_after'].mean():>6.4f}±{sub['global_acc_after'].std():>4.4f} "
              f"{sub['global_l2_auc'].mean():>6.4f}±{sub['global_l2_auc'].std():>4.4f} "
              f"{sub['global_conf_auc'].mean():>6.4f}±{sub['global_conf_auc'].std():>4.4f} "
              f"{sub['mean_cross_l2_auc'].mean():>6.4f}±{sub['mean_cross_l2_auc'].std():>4.4f}")

    # Key observation
    print("\nKey Observations:")
    if float('inf') in epsilons and 1.0 in epsilons:
        baseline = df[df['epsilon'] == float('inf')]
        strong_dp = df[df['epsilon'] == 1.0]
        if len(baseline) > 0 and len(strong_dp) > 0:
            acc_drop = baseline['global_acc_before'].mean() - strong_dp['global_acc_before'].mean()
            auc_drop = baseline['global_l2_auc'].mean() - strong_dp['global_l2_auc'].mean()
            print(f"  ε=1 vs ε=∞: Accuracy drop = {acc_drop:.4f}, L2 AUC drop = {auc_drop:.4f}")
            if acc_drop > 0.1:
                print(f"  → Strong DP (ε=1) causes significant utility loss ({acc_drop:.1%})")
            if auc_drop > 0.1:
                print(f"  → Strong DP reduces attack effectiveness by {auc_drop:.4f}")
            else:
                print(f"  → Even strong DP only marginally reduces geometric leakage")


if __name__ == '__main__':
    main()
