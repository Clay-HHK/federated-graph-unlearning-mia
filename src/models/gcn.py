"""
Graph Convolutional Network (GCN) models for node classification.

This module provides both 2-layer and 3-layer GCN implementations
used across all experiments in the Hub-Ripple MIA research.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN2Layer(torch.nn.Module):
    """
    2-layer Graph Convolutional Network.

    Architecture:
        Input -> GCN -> ReLU -> Dropout -> GCN -> Output

    Args:
        in_channels (int): Number of input features
        out_channels (int): Number of output classes
        hidden_channels (int): Hidden layer dimension (default: 16)
        dropout (float): Dropout rate (default: 0.5)

    Example:
        >>> model = GCN2Layer(in_channels=1433, out_channels=7)
        >>> out = model(data.x, data.edge_index)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 16, dropout: float = 0.5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None,
                return_hidden: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            return_hidden: If True, return penultimate hidden representation

        Returns:
            Node embeddings [num_nodes, out_channels] or hidden [num_nodes, hidden_channels]
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_hidden:
            return x
        x = self.conv2(x, edge_index, edge_weight)
        return x


class GCN3Layer(torch.nn.Module):
    """
    3-layer Graph Convolutional Network.

    Architecture:
        Input -> GCN -> ReLU -> Dropout -> GCN -> ReLU -> Dropout -> GCN -> Output

    Args:
        in_channels (int): Number of input features
        out_channels (int): Number of output classes
        hidden_channels (int): Hidden layer dimension (default: 16)
        dropout (float): Dropout rate (default: 0.5)

    Example:
        >>> model = GCN3Layer(in_channels=1433, out_channels=7)
        >>> out = model(data.x, data.edge_index)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 16, dropout: float = 0.5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None,
                return_hidden: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            return_hidden: If True, return penultimate hidden representation

        Returns:
            Node embeddings [num_nodes, out_channels] or hidden [num_nodes, hidden_channels]
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_hidden:
            return x

        x = self.conv3(x, edge_index, edge_weight)
        return x


# Alias for backward compatibility
GCN = GCN2Layer
