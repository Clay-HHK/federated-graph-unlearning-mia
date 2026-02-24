"""
Federated learning client for graph neural networks.

Each client holds a subgraph (SubgraphResult) and a local GNN model.
Supports local training, model upload/download, embedding extraction,
and node removal for unlearning.
"""

import torch
import torch.nn.functional as F
import copy
from typing import Optional
from collections import OrderedDict
from torch_geometric.data import Data

from .subgraph import SubgraphResult
from ..models.training import get_embeddings, evaluate_model


class FederatedClient:
    """Federated learning client managing a local subgraph and model.

    Args:
        client_id: Client identifier
        subgraph: SubgraphResult containing local data and index mappings
        model: GNN model (will be deep-copied for local training)
        device: Computation device
    """

    def __init__(
        self,
        client_id: int,
        subgraph: SubgraphResult,
        model: torch.nn.Module,
        device: torch.device,
    ):
        self.client_id = client_id
        self.subgraph = subgraph
        self.model = copy.deepcopy(model).to(device)
        self.device = device

    @property
    def num_nodes(self) -> int:
        return self.subgraph.num_nodes

    def train_local(
        self,
        epochs: int = 5,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
    ) -> float:
        """
        Local training on the client's subgraph.

        Args:
            epochs: Number of local training epochs
            lr: Learning rate
            weight_decay: Weight decay for Adam optimizer

        Returns:
            Final training loss
        """
        data = self.subgraph.data.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        loss_val = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)

            # Use train_mask if available, otherwise train on all nodes
            if hasattr(data, 'train_mask') and data.train_mask is not None and data.train_mask.sum() > 0:
                mask = data.train_mask
            else:
                mask = torch.ones(data.num_nodes, dtype=torch.bool, device=self.device)

            loss = F.cross_entropy(out[mask], data.y[mask])
            loss.backward()
            optimizer.step()
            loss_val = loss.item()

        return loss_val

    def upload_model(self) -> OrderedDict:
        """Return model state_dict for server aggregation."""
        return copy.deepcopy(self.model.state_dict())

    def download_model(self, global_state: OrderedDict):
        """Load global model parameters from server."""
        self.model.load_state_dict(copy.deepcopy(global_state))

    def get_embeddings(self, data: Optional[Data] = None) -> torch.Tensor:
        """
        Extract node embeddings.

        Args:
            data: If None, use client's own subgraph data.
                  If provided, extract embeddings on arbitrary data (e.g., full graph).

        Returns:
            Node embeddings tensor
        """
        if data is None:
            data = self.subgraph.data.to(self.device)
        return get_embeddings(self.model, data)

    def evaluate(self, data: Optional[Data] = None) -> float:
        """
        Evaluate model accuracy.

        Args:
            data: If None, use client's own subgraph data.

        Returns:
            Accuracy score
        """
        if data is None:
            data = self.subgraph.data.to(self.device)
        return evaluate_model(self.model, data)

    def remove_node(self, global_node_idx: int) -> Data:
        """
        Remove a target node from the client's subgraph.

        Converts global ID to local ID, then removes all edges
        connected to that node.

        Args:
            global_node_idx: Global node index to remove

        Returns:
            Modified subgraph Data with target's edges removed

        Raises:
            ValueError: If node doesn't belong to this client
        """
        local_idx = self.subgraph.global_to_local(global_node_idx)
        if local_idx is None:
            raise ValueError(
                f"Node {global_node_idx} does not belong to client {self.client_id}"
            )

        data = self.subgraph.data
        row, col = data.edge_index
        mask = (row != local_idx) & (col != local_idx)

        new_data = data.clone()
        new_data.edge_index = data.edge_index[:, mask]

        self.subgraph.data = new_data
        return new_data

    def get_model_state(self) -> OrderedDict:
        """Get a deep copy of the current model state."""
        return copy.deepcopy(self.model.state_dict())

    def load_model_state(self, state_dict: OrderedDict):
        """Load a saved model state."""
        self.model.load_state_dict(copy.deepcopy(state_dict))
