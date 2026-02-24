"""
RQ4: Threat Model Comparison Experiment

Evaluates four threat models with decreasing adversary capability:
- TM1 (White-Box): Full info — global model + all local models + all data
- TM2 (Local Auditor): Participant — global model + own subgraph
- TM3 (Server Auditor): Server — global model weights + API queries
- TM4 (Black-Box): API only — confidence scores only

For TM2, we test from two perspectives:
- TM2-Same: auditor is on the same client as the target (strongest TM2)
- TM2-Cross: auditor is on a different client (practical scenario)

Expected result hierarchy: TM1 > TM3 > TM2-Same > TM2-Cross > TM4

Output columns:
  dataset, method, trial, threat_model, l2_auc, conf_auc,
  available_levels, detail_*

Usage:
    python experiments/federated/run_fed_threat_model.py --trials 100
    python experiments/federated/run_fed_threat_model.py --quick --trials 20
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
from src.federated.unlearning import fed_retrain, fed_gnndelete, fed_gif, fed_graph_eraser
from src.federated.attacks.threat_models import (
    TM1_WhiteBox, TM2_LocalAuditor, TM3_ServerAuditor, TM4_BlackBox,
    evaluate_threat_model,
)


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


def run_trial(data, partition_result, client_subgraphs, model_template,
              device, method, trial_idx, grid, seed=42):
    """Run single trial: train, unlearn, then evaluate all threat models."""
    trial_seed = seed + trial_idx
    set_seed(trial_seed)

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1
    num_clients = grid['num_clients']

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

    # Evaluate all threat models
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
            data, client_subgraphs,
            target_idx, target_client_id,
            hubs, ctrls,
            model_template, device,
        )

        row = {
            'threat_model': tm_name,
            'l2_auc': tm_result.l2_auc,
            'conf_auc': tm_result.conf_auc,
            'available_levels': ','.join(tm_result.available_levels),
            'target_idx': target_idx,
            'target_client_id': target_client_id,
            'num_hubs': len(hubs),
        }
        # Flatten detail dict
        for k, v in tm_result.detail.items():
            row[f'detail_{k}'] = v

        results.append(row)

    return results


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
    print("RQ4: Threat Model Comparison Experiment")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Trials: {args.trials}")
    print(f"Datasets: {grid['datasets']}")
    print(f"Methods: {grid['methods']}")
    print(f"Threat Models: TM1 (WhiteBox), TM2-Same, TM2-Cross, "
          f"TM3 (Server), TM4 (BlackBox)")
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

        for method in grid['methods']:
            config_idx += 1
            print(f"\n  [{config_idx}/{total_configs}] {method}")

            t_start = time.time()
            n_valid = 0

            for trial in range(args.trials):
                tm_results = run_trial(
                    data, partition_result, client_subgraphs,
                    model_template, device, method,
                    trial_idx=trial, grid=grid, seed=args.seed,
                )
                if tm_results is None:
                    continue

                n_valid += 1
                for row in tm_results:
                    row.update({
                        'dataset': dataset_name,
                        'method': method,
                        'trial': trial,
                        'homophily': homophily,
                    })
                    all_results.append(row)

            elapsed = time.time() - t_start
            print(f"    {n_valid} valid trials, {elapsed:.1f}s")

            # Quick summary
            config_df = pd.DataFrame([r for r in all_results
                                       if r['dataset'] == dataset_name
                                       and r['method'] == method])
            if len(config_df) > 0:
                for tm_name in config_df['threat_model'].unique():
                    tm_df = config_df[config_df['threat_model'] == tm_name]
                    print(f"      {tm_name}: L2={tm_df['l2_auc'].mean():.4f}, "
                          f"Conf={tm_df['conf_auc'].mean():.4f}")

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
        output_path = os.path.join(output_dir, f'rq4_threat_model_{timestamp}.csv')

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # ================================================================
    # Summary analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("RQ4 ANALYSIS: Threat Model Comparison")
    print("=" * 70)

    # Table 1: Overall comparison
    print("\n[Table 1] Mean AUC by Threat Model (all datasets, all methods)")
    print(f"{'Threat Model':<20} {'L2 AUC':>10} {'Conf AUC':>10} {'Gap':>8} {'N':>6}")
    print("─" * 56)
    tm_order = ['TM1_WhiteBox', 'TM2_Same', 'TM3_ServerAuditor', 'TM2_Cross', 'TM4_BlackBox']
    for tm_name in tm_order:
        tm_df = df[df['threat_model'] == tm_name]
        if len(tm_df) == 0:
            continue
        l2 = tm_df['l2_auc'].mean()
        conf = tm_df['conf_auc'].mean()
        print(f"{tm_name:<20} {l2:>10.4f} {conf:>10.4f} {l2 - conf:>+8.4f} {len(tm_df):>6}")

    # Table 2: Per dataset
    print("\n[Table 2] L2 AUC by Threat Model × Dataset")
    for dataset_name in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset_name]
        print(f"\n  {dataset_name}:")
        print(f"  {'TM':<20} {'L2 AUC':>10} {'Conf AUC':>10}")
        print(f"  {'─' * 42}")
        for tm_name in tm_order:
            tm_df = ds_df[ds_df['threat_model'] == tm_name]
            if len(tm_df) == 0:
                continue
            print(f"  {tm_name:<20} {tm_df['l2_auc'].mean():>10.4f} "
                  f"{tm_df['conf_auc'].mean():>10.4f}")

    # Table 3: Per method
    print("\n[Table 3] L2 AUC by Threat Model × Method")
    for method in df['method'].unique():
        m_df = df[df['method'] == method]
        print(f"\n  {method}:")
        print(f"  {'TM':<20} {'L2 AUC':>10} {'Conf AUC':>10}")
        print(f"  {'─' * 42}")
        for tm_name in tm_order:
            tm_df = m_df[m_df['threat_model'] == tm_name]
            if len(tm_df) == 0:
                continue
            print(f"  {tm_name:<20} {tm_df['l2_auc'].mean():>10.4f} "
                  f"{tm_df['conf_auc'].mean():>10.4f}")

    # Verification
    print("\n" + "=" * 70)
    print("RQ4 Verification")
    print("=" * 70)

    tm_means = {}
    for tm_name in tm_order:
        tm_df = df[df['threat_model'] == tm_name]
        if len(tm_df) > 0:
            tm_means[tm_name] = tm_df['l2_auc'].mean()

    if len(tm_means) >= 4:
        # Check ordering
        ordering_parts = []
        for tm_name in tm_order:
            if tm_name in tm_means:
                ordering_parts.append(f"{tm_name}({tm_means[tm_name]:.3f})")
        print(f"  Ordering: {' > '.join(ordering_parts)}")

        # Check TM4 is near chance
        if 'TM4_BlackBox' in tm_means:
            tm4 = tm_means['TM4_BlackBox']
            print(f"  TM4 L2 AUC = {tm4:.4f} (expected: 0.5, since L2 not accessible)")
            print(f"  TM4 effectively blind to geometric leakage: "
                  f"{'CONFIRMED' if abs(tm4 - 0.5) < 0.05 else 'UNEXPECTED'}")

        # Check TM1 is strongest
        if 'TM1_WhiteBox' in tm_means:
            tm1 = tm_means['TM1_WhiteBox']
            others = [v for k, v in tm_means.items() if k != 'TM1_WhiteBox']
            if others and tm1 >= max(others):
                print(f"  TM1 strongest adversary: CONFIRMED")
            else:
                print(f"  TM1 strongest adversary: UNEXPECTED (some TM is higher)")


if __name__ == '__main__':
    main()
