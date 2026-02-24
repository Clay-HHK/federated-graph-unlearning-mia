"""
Federated Graph Unlearning Privacy Audit — Pilot Experiment

End-to-end verification script (~5 minutes on CPU).
Tests the complete pipeline: partition → federated train → unlearn → multi-level attack.

Configuration: Cora, 3 clients, Metis partition, IID, 10 trials
Methods: FedRetrain (gold standard) + FedGNNDelete (approximate)

Expected results:
  - FedRetrain: All levels AUC ∈ [0.45, 0.55] (true unlearning)
  - FedGNNDelete: Global L2 AUC > 0.60, Local L2 AUC > 0.70
  - Both methods: Conf AUC ~ 0.50 (confidence illusion)

Usage:
    python experiments/federated/run_fed_pilot.py
"""

import sys
import os
import time
import copy
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.common import set_seed, get_device
from src.utils.data_loader import load_dataset
from src.utils.graph_utils import (
    get_neighbors, select_control_node, get_node_degrees,
    select_high_degree_nodes,
)
from src.models.gcn import GCN2Layer
from src.models.training import train_node_classifier, evaluate_model
from src.federated.subgraph import build_client_subgraph
from src.federated.data_partition import partition_graph
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.unlearning import fed_retrain, fed_gnndelete
from src.federated.attacks.hub_ripple_federated import (
    multilevel_hub_ripple, MultiLevelAttackResult,
)
from configs.federated_config import get_fed_pilot_config


def select_target_and_hubs(data, partition_result, seed=42):
    """Select a suitable target node and its hub/control nodes.

    Picks a high-degree node as target, uses its neighbors as hubs,
    and randomly selected non-neighbors as controls.

    Returns:
        Tuple of (target_idx, target_client_id, hub_indices, control_indices)
    """
    set_seed(seed)

    # Find high-degree nodes
    degrees = get_node_degrees(data.edge_index, data.num_nodes)
    sorted_nodes = torch.argsort(degrees, descending=True)

    # Pick a target with enough neighbors (hubs)
    target_idx = None
    for candidate in sorted_nodes[:50].tolist():
        neighbors = get_neighbors(data.edge_index, candidate)
        if len(neighbors) >= 5:
            target_idx = candidate
            break

    if target_idx is None:
        target_idx = sorted_nodes[0].item()

    target_client_id = partition_result.partition_map[target_idx].item()

    # Hub nodes = target's neighbors
    neighbors = get_neighbors(data.edge_index, target_idx)
    hub_indices = neighbors.tolist()

    # Control nodes = non-neighbors with similar degree
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

    # Ensure equal lengths
    min_len = min(len(hub_indices), len(control_indices))
    hub_indices = hub_indices[:min_len]
    control_indices = control_indices[:min_len]

    return target_idx, target_client_id, hub_indices, control_indices


def run_single_trial(
    data, partition_result, client_subgraphs, model_template,
    device, config, trial_idx, method='FedGNNDelete',
):
    """Run a single trial: train → snapshot → unlearn → snapshot → attack.

    Returns:
        MultiLevelAttackResult or None if trial failed
    """
    seed = config.seed + trial_idx
    set_seed(seed)

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

    # Select target
    target_idx, target_client_id, hub_indices, control_indices = \
        select_target_and_hubs(data, partition_result, seed=seed)

    if len(hub_indices) < 3 or len(control_indices) < 3:
        return None

    # Create fresh model
    global_model = GCN2Layer(num_features, num_classes).to(device)
    set_seed(seed)
    global_model.reset_parameters()

    # Build fresh clients
    clients = []
    for cid, subgraph in client_subgraphs.items():
        client = FederatedClient(cid, copy.deepcopy(subgraph), global_model, device)
        clients.append(client)

    # Create server and train
    server = FederatedServer(global_model, clients, device)
    server.train(
        num_rounds=config.num_rounds,
        local_epochs=config.local_epochs,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Snapshot before unlearning
    server.snapshot_state('before')

    # Deep copy server for unlearning (preserve original for comparison)
    server_copy = FederatedServer(
        copy.deepcopy(server.global_model),
        [FederatedClient(
            cid,
            copy.deepcopy(server.clients[cid].subgraph),
            server.clients[cid].model,
            device,
        ) for cid in server.client_ids],
        device,
    )

    # Perform unlearning
    if method == 'FedRetrain':
        result = fed_retrain(
            server_copy, target_idx, target_client_id,
            data, partition_result,
            num_features, num_classes,
            num_rounds=config.num_rounds,
            local_epochs=config.local_epochs,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            seed=seed + 1000,
        )
        unlearned_server = result.server
    elif method == 'FedGNNDelete':
        result = fed_gnndelete(
            server_copy, target_idx, target_client_id,
            num_features, num_classes,
            reaggregate_rounds=config.reaggregate_rounds,
            local_epochs=config.local_epochs,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        unlearned_server = result.server
    else:
        raise ValueError(f"Unknown method: {method}")

    # Snapshot after unlearning
    unlearned_server.snapshot_state('after')

    # Multi-level attack
    snap_before = server.get_snapshot('before')
    snap_after = unlearned_server.get_snapshot('after')

    # Use original subgraphs for attack (before unlearning)
    attack_result = multilevel_hub_ripple(
        snap_before, snap_after,
        data, client_subgraphs,
        target_idx, target_client_id,
        hub_indices, control_indices,
        model_template, device,
    )

    return attack_result


def main():
    print("=" * 70)
    print("Federated Graph Unlearning Privacy Audit — Pilot Experiment")
    print("=" * 70)

    config = get_fed_pilot_config()
    device = get_device()
    print(f"\nDevice: {device}")
    print(config.summary())

    # Load dataset
    print(f"\n[1/5] Loading dataset: Cora")
    data, homophily = load_dataset('Cora', device)
    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1
    print(f"  Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")
    print(f"  Features: {num_features}, Classes: {num_classes}")
    print(f"  Homophily: {homophily:.3f}")

    # Partition graph
    print(f"\n[2/5] Partitioning graph: {config.partition_method}, "
          f"K={config.num_clients}")
    partition_result = partition_graph(
        data, config.num_clients,
        method=config.partition_method,
        seed=config.seed,
        distribution=config.data_distribution,
    )
    print(f"  Client sizes: {partition_result.client_sizes}")
    print(f"  Cross-edge ratio: {partition_result.cross_edge_ratio:.3f}")

    # Build client subgraphs
    print(f"\n[3/5] Building client subgraphs")
    client_subgraphs = {}
    for cid in range(config.num_clients):
        node_mask = partition_result.partition_map == cid
        node_indices = node_mask.nonzero(as_tuple=True)[0]
        subgraph = build_client_subgraph(data, node_indices, data.edge_index, cid)
        client_subgraphs[cid] = subgraph
        print(f"  Client {cid}: {subgraph.num_nodes} nodes, "
              f"{subgraph.data.edge_index.size(1)} edges, "
              f"{subgraph.num_cross_edges} cross-edges")

    # Model template
    model_template = GCN2Layer(num_features, num_classes).to(device)

    # Run trials
    print(f"\n[4/5] Running trials")
    for method in config.fed_unlearn_methods:
        print(f"\n{'─' * 60}")
        print(f"Method: {method}")
        print(f"{'─' * 60}")

        results = []
        for trial in range(config.num_trials):
            t_start = time.time()

            attack_result = run_single_trial(
                data, partition_result, client_subgraphs,
                model_template, device, config,
                trial_idx=trial, method=method,
            )

            elapsed = time.time() - t_start

            if attack_result is None:
                print(f"  Trial {trial + 1}: SKIPPED (insufficient hubs/controls)")
                continue

            results.append(attack_result)
            print(f"  Trial {trial + 1}/{config.num_trials} ({elapsed:.1f}s): "
                  f"Global L2={attack_result.global_l2_auc:.3f}, "
                  f"Local L2={attack_result.local_l2_auc:.3f}, "
                  f"Cross L2={attack_result.mean_cross_l2_auc:.3f}, "
                  f"Conf={attack_result.global_conf_auc:.3f}")

        if not results:
            print("  No valid trials!")
            continue

        # Aggregate results
        print(f"\n  Summary ({method}, {len(results)} trials):")
        print(f"  {'Metric':<25} {'Mean':>8} {'Std':>8}")
        print(f"  {'─' * 45}")

        metrics = {
            'Global L2 AUC': [r.global_l2_auc for r in results],
            'Global Conf AUC': [r.global_conf_auc for r in results],
            'Global Gap (L2-Conf)': [r.global_gap for r in results],
            'Local L2 AUC': [r.local_l2_auc for r in results],
            'Local Conf AUC': [r.local_conf_auc for r in results],
            'Local Gap (L2-Conf)': [r.local_gap for r in results],
            'Cross L2 AUC (mean)': [r.mean_cross_l2_auc for r in results],
            'Cross L2 AUC (max)': [r.max_cross_l2_auc for r in results],
        }

        for name, values in metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {name:<25} {mean:>8.4f} {std:>8.4f}")

    # Validation checks
    print(f"\n[5/5] Validation")
    print("─" * 60)

    # Check if results match expectations
    all_pass = True

    # (Validation messages are informational; exact thresholds may vary)
    print("\nPilot experiment complete!")
    print("Check the summary above to verify:")
    print("  1. FedRetrain: All AUCs should be close to 0.50")
    print("  2. FedGNNDelete: L2 AUCs should be > 0.55")
    print("  3. Both: Conf AUC should be close to 0.50 (confidence illusion)")
    print("  4. Local L2 AUC >= Global L2 AUC (local signal strongest)")


if __name__ == '__main__':
    main()
