"""
Federated graph unlearning methods.

Implements four federated unlearning strategies:
1. FedRetrain: Gold standard — remove target, retrain entire federation from scratch
2. FedGNNDelete: Gradient-based local unlearning on affected client + re-aggregation
3. FedGraphEraser: SISA-based local unlearning within affected client + re-aggregation
4. FedClientUnlearning: Entire client departure + re-aggregation of remaining clients
"""

import time
import copy
import torch
from dataclasses import dataclass
from typing import Optional
from torch_geometric.data import Data

from .server import FederatedServer
from .client import FederatedClient
from .subgraph import SubgraphResult, build_client_subgraph
from .data_partition import PartitionResult
from ..models.gcn import GCN2Layer
from ..models.training import train_node_classifier
from ..unlearning.gnn_delete import unlearn_gnndelete
from ..unlearning.gif import unlearn_gif
from ..unlearning.sisa import partition_bekm, partition_blpa, build_shard_data
from ..utils.common import set_seed


@dataclass
class FedUnlearnResult:
    """Federated unlearning result.

    Attributes:
        server: Updated FederatedServer after unlearning
        unlearn_time: Wall-clock time in seconds
        method: Unlearning method name
        target_client_id: Client that owned the target node
        reaggregate_rounds: Number of re-aggregation rounds used
    """
    server: FederatedServer
    unlearn_time: float
    method: str
    target_client_id: int
    reaggregate_rounds: int


def fed_retrain(
    server: FederatedServer,
    target_idx: int,
    target_client_id: int,
    full_data: Data,
    partition_result: PartitionResult,
    num_features: int,
    num_classes: int,
    num_rounds: int = 50,
    local_epochs: int = 5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    seed: int = 42,
) -> FedUnlearnResult:
    """
    Federated retraining (gold standard).

    Removes target node from affected client's subgraph, then retrains
    the entire federation from random initialization.

    Expected: All audit levels should yield AUC ~ 0.5 (true unlearning).
    """
    start = time.time()

    device = full_data.x.device

    # Create a fresh global model
    global_model = GCN2Layer(num_features, num_classes).to(device)
    set_seed(seed)
    global_model.reset_parameters()

    # Rebuild all client subgraphs, removing target from its client
    new_clients = []
    for cid in sorted(partition_result.partition_map.unique().tolist()):
        node_mask = partition_result.partition_map == cid
        node_indices = node_mask.nonzero(as_tuple=True)[0]

        # Remove target node from its client
        if cid == target_client_id:
            node_indices = node_indices[node_indices != target_idx]

        if len(node_indices) == 0:
            continue

        # Remove target's edges from full graph
        row, col = full_data.edge_index
        edge_mask = (row != target_idx) & (col != target_idx)
        clean_edge_index = full_data.edge_index[:, edge_mask]

        subgraph = build_client_subgraph(
            full_data, node_indices, clean_edge_index, cid
        )
        client = FederatedClient(cid, subgraph, global_model, device)
        new_clients.append(client)

    # Create new server and train from scratch
    new_server = FederatedServer(global_model, new_clients, device)
    new_server.train(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr,
        weight_decay=weight_decay,
    )

    elapsed = time.time() - start
    return FedUnlearnResult(
        server=new_server,
        unlearn_time=elapsed,
        method='FedRetrain',
        target_client_id=target_client_id,
        reaggregate_rounds=num_rounds,
    )


def fed_gnndelete(
    server: FederatedServer,
    target_idx: int,
    target_client_id: int,
    num_features: int,
    num_classes: int,
    reaggregate_rounds: int = 5,
    local_epochs: int = 5,
    unlearn_steps: int = 20,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> FedUnlearnResult:
    """
    Federated GNNDelete unlearning.

    Only the affected client performs gradient-based GNNDelete locally.
    Then re-aggregation rounds propagate the update to the global model.

    Expected: Local L2 AUC >> 0.5, Global L2 AUC > 0.5, Conf AUC ~ 0.5.
    """
    start = time.time()

    device = server.device
    target_client = server.clients[target_client_id]
    subgraph = target_client.subgraph
    local_data = subgraph.data.to(device)

    # Convert global target_idx to local
    local_target = subgraph.global_to_local(target_idx)
    if local_target is None:
        raise ValueError(
            f"Target node {target_idx} not found in client {target_client_id}"
        )

    # Perform GNNDelete on local model
    unlearned_model = unlearn_gnndelete(
        model=target_client.model,
        data=local_data,
        target_idx=local_target,
        num_features=num_features,
        num_classes=num_classes,
        device=device,
        steps=unlearn_steps,
        learning_rate=lr,
    )

    # Update the client's model
    target_client.model = unlearned_model

    # Also remove target's edges from subgraph
    target_client.remove_node(target_idx)

    # Re-aggregation rounds to propagate changes
    if reaggregate_rounds > 0:
        server.reaggregate(
            num_rounds=reaggregate_rounds,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
        )

    elapsed = time.time() - start
    return FedUnlearnResult(
        server=server,
        unlearn_time=elapsed,
        method='FedGNNDelete',
        target_client_id=target_client_id,
        reaggregate_rounds=reaggregate_rounds,
    )


def fed_gif(
    server: FederatedServer,
    target_idx: int,
    target_client_id: int,
    num_features: int,
    num_classes: int,
    reaggregate_rounds: int = 5,
    local_epochs: int = 5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> FedUnlearnResult:
    """
    Federated GIF (Graph Influence Function) unlearning.

    The affected client performs influence-function-based unlearning locally,
    then re-aggregation propagates the update to the global model.

    GIF is a one-step method: it computes the gradient on the target node
    and removes its influence via a single parameter update.
    """
    start = time.time()

    device = server.device
    target_client = server.clients[target_client_id]
    subgraph = target_client.subgraph
    local_data = subgraph.data.to(device)

    local_target = subgraph.global_to_local(target_idx)
    if local_target is None:
        raise ValueError(
            f"Target node {target_idx} not found in client {target_client_id}"
        )

    # Perform GIF on local model
    unlearned_model = unlearn_gif(
        model=target_client.model,
        data=local_data,
        target_idx=local_target,
        num_features=num_features,
        num_classes=num_classes,
        device=device,
        learning_rate=lr,
    )

    target_client.model = unlearned_model
    target_client.remove_node(target_idx)

    if reaggregate_rounds > 0:
        server.reaggregate(
            num_rounds=reaggregate_rounds,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
        )

    elapsed = time.time() - start
    return FedUnlearnResult(
        server=server,
        unlearn_time=elapsed,
        method='FedGIF',
        target_client_id=target_client_id,
        reaggregate_rounds=reaggregate_rounds,
    )


def fed_graph_eraser(
    server: FederatedServer,
    target_idx: int,
    target_client_id: int,
    num_features: int,
    num_classes: int,
    reaggregate_rounds: int = 5,
    local_epochs: int = 5,
    num_local_shards: int = 3,
    partition_strategy: str = 'bekm',
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 100,
    seed: int = 42,
) -> FedUnlearnResult:
    """
    Federated GraphEraser unlearning.

    The affected client internally uses SISA-style shard-based unlearning:
    1. Partition client's subgraph into local shards (BEKM or BLPA)
    2. Identify which shard contains the target
    3. Retrain only that shard (excluding target)
    4. Re-aggregate locally within the client, then federated re-aggregation

    Args:
        partition_strategy: 'bekm' or 'blpa' for local shard partitioning
    """
    start = time.time()

    device = server.device
    target_client = server.clients[target_client_id]
    subgraph = target_client.subgraph
    local_data = subgraph.data.to(device)

    local_target = subgraph.global_to_local(target_idx)
    if local_target is None:
        raise ValueError(
            f"Target node {target_idx} not found in client {target_client_id}"
        )

    # Partition client's subgraph into local shards
    if partition_strategy == 'blpa':
        shard_map = partition_blpa(
            local_data,
            num_shards=num_local_shards,
            seed=seed,
        )
    else:
        shard_map = partition_bekm(
            local_data,
            num_shards=num_local_shards,
            seed=seed,
        )

    # Find target's shard
    target_shard = shard_map[local_target].item()

    # Train all shard models (following GraphEraser paper: Chen et al., CCS 2022)
    # Each shard gets an independent model; the affected shard excludes the target
    shard_states = []
    shard_sizes = []
    for sid in range(num_local_shards):
        exclude = local_target if sid == target_shard else None
        shard_data = build_shard_data(
            local_data, shard_map, sid, exclude_node=exclude
        )
        shard_model = GCN2Layer(num_features, num_classes).to(device)
        set_seed(seed + sid)
        shard_model.reset_parameters()

        opt = torch.optim.Adam(
            shard_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        train_node_classifier(shard_model, shard_data, opt, epochs=epochs)

        shard_states.append(shard_model.state_dict())
        shard_sizes.append(shard_data.num_nodes)

    # Aggregate shard models via weighted averaging (proportional to shard size)
    # This follows GraphEraser's aggregation step, approximated as FedAvg-style
    # weight averaging instead of learned importance scores for simplicity
    total_nodes = sum(shard_sizes)
    aggregated_state = {}
    for key in shard_states[0]:
        aggregated_state[key] = sum(
            shard_states[i][key] * (shard_sizes[i] / total_nodes)
            for i in range(num_local_shards)
        )

    # Create aggregated client model
    client_model = GCN2Layer(num_features, num_classes).to(device)
    client_model.load_state_dict(aggregated_state)
    target_client.model = client_model

    # Remove target edges
    target_client.remove_node(target_idx)

    # Federated re-aggregation
    if reaggregate_rounds > 0:
        server.reaggregate(
            num_rounds=reaggregate_rounds,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
        )

    method_name = f'FedGraphEraser-{partition_strategy.upper()}'
    elapsed = time.time() - start
    return FedUnlearnResult(
        server=server,
        unlearn_time=elapsed,
        method=method_name,
        target_client_id=target_client_id,
        reaggregate_rounds=reaggregate_rounds,
    )


def fed_client_unlearning(
    server: FederatedServer,
    departing_client_id: int,
    reaggregate_rounds: int = 5,
    local_epochs: int = 5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> FedUnlearnResult:
    """
    Client-level unlearning: entire client departs the federation.

    1. Remove departing client from the federation
    2. Re-aggregate remaining K-1 clients
    3. Optional: additional training rounds

    Expected: Better privacy than node-level (removes all client data),
    but cross-client leakage may persist through shared global model history.
    """
    start = time.time()

    # Remove client
    server.remove_client(departing_client_id)

    # Re-aggregate remaining clients
    server.reaggregate_remaining()

    # Additional training rounds
    if reaggregate_rounds > 0:
        server.reaggregate(
            num_rounds=reaggregate_rounds,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
        )

    elapsed = time.time() - start
    return FedUnlearnResult(
        server=server,
        unlearn_time=elapsed,
        method='FedClientUnlearning',
        target_client_id=departing_client_id,
        reaggregate_rounds=reaggregate_rounds,
    )
