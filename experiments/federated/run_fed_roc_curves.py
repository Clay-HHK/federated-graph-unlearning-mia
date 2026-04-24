"""
Experiment C: ROC Curve Generation (Reviewer #7)

Generates full ROC curves for representative configurations and
reports detection rates at multiple thresholds {0.55, 0.60, 0.65, 0.70}.

Configs: Cora × {FedGNNDelete, FedRetrain} + Squirrel × {FedGNNDelete, FedRetrain}

Usage:
    python experiments/federated/run_fed_roc_curves.py --trials 50
"""

import sys
import os
import time
import copy
import json
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
from src.models.training import get_embeddings
from src.federated.subgraph import build_client_subgraph
from src.federated.data_partition import partition_graph
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.unlearning import fed_retrain, fed_gnndelete
from src.federated.attacks.hub_ripple_federated import multilevel_hub_ripple
from src.attacks.hub_ripple import measure_embedding_drift

from sklearn.metrics import roc_curve, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description='ROC Curve Generation')
    parser.add_argument('--trials', type=int, default=50)
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


def run_trial_with_roc(data, partition_result, client_subgraphs,
                       model_template, device, method, trial_idx, grid, seed=42):
    """Run trial and return both scalar AUC and full ROC data."""
    trial_seed = seed + trial_idx
    set_seed(trial_seed)

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

    target_idx, target_client_id, hubs, ctrls = \
        select_target_and_hubs(data, partition_result, seed=trial_seed)

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

    server.snapshot_state('before')

    ul_clients = []
    for cid in server.client_ids:
        ul_clients.append(FederatedClient(
            cid, copy.deepcopy(server.clients[cid].subgraph),
            server.clients[cid].model, device,
        ))
    ul_server = FederatedServer(copy.deepcopy(server.global_model), ul_clients, device)

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

    ul_server = result.server
    ul_server.snapshot_state('after')

    # Compute raw drift scores for ROC
    snap_before = server.get_snapshot('before')
    snap_after = ul_server.get_snapshot('after')

    # Global level: compute per-node L2 drift
    data_dev = data.to(device)
    global_model_before = GCN2Layer(num_features, num_classes).to(device)
    global_model_before.load_state_dict(snap_before.global_state)
    global_model_after = GCN2Layer(num_features, num_classes).to(device)
    global_model_after.load_state_dict(snap_after.global_state)

    emb_before = get_embeddings(global_model_before, data_dev)
    emb_after = get_embeddings(global_model_after, data_dev)

    hub_drifts = []
    ctrl_drifts = []
    for h in hubs:
        drift = measure_embedding_drift(emb_before[h], emb_after[h], metric='l2')
        hub_drifts.append(drift)
    for c in ctrls:
        drift = measure_embedding_drift(emb_before[c], emb_after[c], metric='l2')
        ctrl_drifts.append(drift)

    y_true = [1] * len(hub_drifts) + [0] * len(ctrl_drifts)
    y_scores = hub_drifts + ctrl_drifts

    if len(set(y_true)) < 2:
        return None

    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Also get the multilevel result for scalar metrics
    attack = multilevel_hub_ripple(
        snap_before, snap_after,
        data, client_subgraphs,
        target_idx, target_client_id,
        hubs, ctrls,
        model_template, device,
    )

    return {
        'trial': trial_idx,
        'global_l2_auc': attack.global_l2_auc,
        'global_conf_auc': attack.global_conf_auc,
        'local_l2_auc': attack.local_l2_auc,
        'mean_cross_l2_auc': attack.mean_cross_l2_auc,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'num_hubs': len(hubs),
    }


def main():
    args = parse_args()
    device = get_device()

    grid = {
        'num_rounds': 50,
        'local_epochs': 5,
    }

    configs = [
        ('Cora', 'FedGNNDelete'),
        ('Cora', 'FedRetrain'),
        ('Squirrel', 'FedGNNDelete'),
        ('Squirrel', 'FedRetrain'),
    ]

    print("=" * 70)
    print("Experiment C: ROC Curve Generation")
    print("=" * 70)
    print(f"Configs: {configs}")
    print(f"Trials: {args.trials}")
    print("=" * 70)

    all_roc_data = []
    detection_rates = []

    for dataset_name, method in configs:
        print(f"\n{'━' * 60}")
        print(f"Config: {dataset_name} × {method}")
        print(f"{'━' * 60}")

        data, homophily = load_dataset(dataset_name, device)
        num_features = data.num_features
        num_classes = int(data.y.max().item()) + 1
        model_template = GCN2Layer(num_features, num_classes).to(device)

        partition_result = partition_graph(
            data, 5, method='metis', seed=args.seed,
        )

        client_subgraphs = {}
        for cid in range(5):
            nodes = (partition_result.partition_map == cid).nonzero(as_tuple=True)[0]
            client_subgraphs[cid] = build_client_subgraph(
                data, nodes, data.edge_index, cid
            )

        trial_aucs = []
        for trial in range(args.trials):
            res = run_trial_with_roc(
                data, partition_result, client_subgraphs,
                model_template, device, method,
                trial_idx=trial, grid=grid, seed=args.seed,
            )
            if res is None:
                continue

            all_roc_data.append({
                'dataset': dataset_name,
                'method': method,
                'trial': res['trial'],
                'global_l2_auc': res['global_l2_auc'],
                'fpr': res['fpr'],
                'tpr': res['tpr'],
            })
            trial_aucs.append(res['global_l2_auc'])

        if not trial_aucs:
            continue

        # Detection rates at multiple thresholds
        for threshold in [0.55, 0.60, 0.65, 0.70]:
            rate = np.mean([a > threshold for a in trial_aucs])
            detection_rates.append({
                'dataset': dataset_name,
                'method': method,
                'threshold': threshold,
                'detection_rate': rate,
                'n_trials': len(trial_aucs),
            })

        print(f"  {len(trial_aucs)} valid trials, "
              f"mean AUC = {np.mean(trial_aucs):.4f} ± {np.std(trial_aucs):.4f}")

    # Save ROC data as JSON
    output_dir = 'results/federated/tables'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    roc_path = os.path.join(output_dir, f'roc_data_{timestamp}.json')
    with open(roc_path, 'w') as f:
        json.dump(all_roc_data, f)
    print(f"\nROC data saved to: {roc_path}")

    # Save detection rates as CSV
    dr_df = pd.DataFrame(detection_rates)
    dr_path = os.path.join(output_dir, f'roc_detection_rates_{timestamp}.csv')
    dr_df.to_csv(dr_path, index=False)
    print(f"Detection rates saved to: {dr_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Detection Rate Summary")
    print("=" * 70)
    print(f"{'Dataset':<12} {'Method':<18} {'AUC>0.55':>10} {'AUC>0.60':>10} "
          f"{'AUC>0.65':>10} {'AUC>0.70':>10}")
    print("-" * 72)

    for dataset_name, method in configs:
        sub = dr_df[(dr_df['dataset'] == dataset_name) & (dr_df['method'] == method)]
        if len(sub) == 0:
            continue
        rates = {row['threshold']: row['detection_rate'] for _, row in sub.iterrows()}
        print(f"{dataset_name:<12} {method:<18} "
              f"{rates.get(0.55, 0):>9.1%} {rates.get(0.60, 0):>9.1%} "
              f"{rates.get(0.65, 0):>9.1%} {rates.get(0.70, 0):>9.1%}")


if __name__ == '__main__':
    main()
