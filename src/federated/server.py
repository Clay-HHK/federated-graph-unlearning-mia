"""
Federated server implementing FedAvg aggregation and system state management.

Manages global model, coordinates client training rounds, and provides
snapshot mechanism for before/after unlearning comparisons.
"""

import torch
import time
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import OrderedDict
from torch_geometric.data import Data

from .client import FederatedClient
from ..models.training import get_embeddings, evaluate_model


@dataclass
class SystemSnapshot:
    """Complete system state for before/after comparison.

    Stores deep copies of global and all local model states,
    enabling temporal embedding drift analysis at all audit levels.

    Attributes:
        global_state: Global model state_dict
        local_states: Dict mapping client_id -> local model state_dict
        timestamp: When the snapshot was taken
        metadata: Additional info (round number, accuracies, etc.)
    """
    global_state: OrderedDict
    local_states: Dict[int, OrderedDict]
    timestamp: float
    metadata: Dict = field(default_factory=dict)


class FederatedServer:
    """Federated server managing global model and client coordination.

    Args:
        global_model: GNN model architecture (will be deep-copied)
        clients: List of FederatedClient instances
        device: Computation device
    """

    def __init__(
        self,
        global_model: torch.nn.Module,
        clients: List[FederatedClient],
        device: torch.device,
    ):
        self.global_model = copy.deepcopy(global_model).to(device)
        self.clients = {c.client_id: c for c in clients}
        self.device = device
        self._snapshots: Dict[str, SystemSnapshot] = {}

    @property
    def num_clients(self) -> int:
        return len(self.clients)

    @property
    def client_ids(self) -> List[int]:
        return sorted(self.clients.keys())

    def fedavg_aggregate(
        self,
        client_states: List[OrderedDict],
        client_weights: List[float],
    ) -> OrderedDict:
        """
        Weighted FedAvg aggregation.

        new_param = sum(w_k * param_k) for each parameter.

        Args:
            client_states: List of client model state_dicts
            client_weights: Aggregation weights (typically n_k / sum(n_k))

        Returns:
            Aggregated global model state_dict
        """
        # Normalize weights
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]

        # Weighted average of parameters
        avg_state = OrderedDict()
        for key in client_states[0].keys():
            avg_state[key] = sum(
                w * state[key].float() for w, state in zip(weights, client_states)
            ).to(client_states[0][key].dtype)

        return avg_state

    def broadcast_global_model(self):
        """Send global model to all clients."""
        global_state = self.global_model.state_dict()
        for client in self.clients.values():
            client.download_model(global_state)

    def train(
        self,
        num_rounds: int = 50,
        local_epochs: int = 5,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        verbose: bool = False,
    ) -> List[float]:
        """
        Complete federated training loop.

        Each round: broadcast -> local train -> upload -> aggregate.

        Args:
            num_rounds: Number of federated rounds
            local_epochs: Local training epochs per round
            lr: Client learning rate
            weight_decay: Client weight decay
            verbose: Print per-round info

        Returns:
            List of per-round average training losses
        """
        round_losses = []

        for round_idx in range(num_rounds):
            # 1. Broadcast global model
            self.broadcast_global_model()

            # 2. Local training
            client_states = []
            client_weights = []
            round_loss = 0.0

            for cid in self.client_ids:
                client = self.clients[cid]
                loss = client.train_local(
                    epochs=local_epochs, lr=lr, weight_decay=weight_decay
                )
                client_states.append(client.upload_model())
                client_weights.append(float(client.num_nodes))
                round_loss += loss

            round_loss /= self.num_clients
            round_losses.append(round_loss)

            # 3. Aggregate
            avg_state = self.fedavg_aggregate(client_states, client_weights)
            self.global_model.load_state_dict(avg_state)

            if verbose and (round_idx + 1) % 10 == 0:
                print(f"  Round {round_idx + 1}/{num_rounds}, "
                      f"Avg Loss: {round_loss:.4f}")

        # Final broadcast so clients have the trained model
        self.broadcast_global_model()

        return round_losses

    def reaggregate(
        self,
        num_rounds: int = 5,
        local_epochs: int = 5,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
    ) -> List[float]:
        """
        Re-aggregation rounds after local unlearning.

        Same as train() but typically fewer rounds to propagate
        the unlearning effect to the global model.

        Args:
            num_rounds: Number of re-aggregation rounds
            local_epochs: Local epochs per round
            lr: Learning rate
            weight_decay: Weight decay

        Returns:
            List of per-round losses
        """
        return self.train(
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
        )

    def snapshot_state(self, tag: str):
        """
        Save complete system state (deep copy) for later comparison.

        Stores global model + all client local models.

        Args:
            tag: Identifier for this snapshot (e.g., 'before_unlearning')
        """
        local_states = {}
        for cid, client in self.clients.items():
            local_states[cid] = client.get_model_state()

        self._snapshots[tag] = SystemSnapshot(
            global_state=copy.deepcopy(self.global_model.state_dict()),
            local_states=local_states,
            timestamp=time.time(),
            metadata={'tag': tag, 'num_clients': self.num_clients},
        )

    def get_snapshot(self, tag: str) -> SystemSnapshot:
        """Retrieve a previously saved snapshot."""
        if tag not in self._snapshots:
            raise KeyError(f"Snapshot '{tag}' not found. Available: {list(self._snapshots.keys())}")
        return self._snapshots[tag]

    def evaluate_global(self, full_data: Data) -> float:
        """Evaluate global model accuracy on the full graph."""
        full_data = full_data.to(self.device)
        return evaluate_model(self.global_model, full_data)

    def get_global_embeddings(self, data: Data) -> torch.Tensor:
        """Extract embeddings using the global model."""
        data = data.to(self.device)
        return get_embeddings(self.global_model, data)

    def remove_client(self, client_id: int):
        """Remove a client from the federation (for client-level unlearning)."""
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not found")
        del self.clients[client_id]

    def reaggregate_remaining(self):
        """Re-aggregate using remaining clients (after client removal)."""
        client_states = []
        client_weights = []
        for cid in self.client_ids:
            client = self.clients[cid]
            client_states.append(client.upload_model())
            client_weights.append(float(client.num_nodes))

        avg_state = self.fedavg_aggregate(client_states, client_weights)
        self.global_model.load_state_dict(avg_state)
        self.broadcast_global_model()
