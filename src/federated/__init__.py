"""
Federated Graph Learning module for Multi-Level Privacy Audit.

Core thesis: FedAvg creates a stronger privacy illusion by diluting
confidence-level signals, but L2 geometric audit reveals persistent
leakage at global, local, and cross-client levels.
"""

from .subgraph import SubgraphResult, build_client_subgraph
from .data_partition import PartitionResult, partition_graph
from .client import FederatedClient
from .server import FederatedServer, SystemSnapshot

__all__ = [
    'SubgraphResult',
    'build_client_subgraph',
    'PartitionResult',
    'partition_graph',
    'FederatedClient',
    'FederatedServer',
    'SystemSnapshot',
]
