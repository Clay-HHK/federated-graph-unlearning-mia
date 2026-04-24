"""
Subgraph construction and global-local index mapping for federated graph learning.

Key challenge: each client holds a subgraph with local indices (0..n_k-1),
but the attack needs global-level embedding comparisons. SubgraphResult
maintains bidirectional index mapping to bridge this gap.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Optional
from torch_geometric.data import Data

from ..utils.graph_utils import filter_edges_by_nodes_vectorized, remap_edge_indices


@dataclass
class SubgraphResult:
    """Client subgraph with bidirectional index mapping.

    Attributes:
        data: PyG subgraph with local node indices (0..n_k-1)
        global_ids: Tensor mapping local_idx -> global_idx, shape [n_k]
        local_map: Dict mapping global_idx -> local_idx
        client_id: Client identifier
        cross_edges: Cross-client edges involving this client's nodes (global indices), shape [2, num_cross]
    """
    data: Data
    global_ids: torch.Tensor
    local_map: Dict[int, int]
    client_id: int
    cross_edges: torch.Tensor

    def global_to_local(self, global_idx: int) -> Optional[int]:
        """Convert global node index to local index. Returns None if not in this client."""
        return self.local_map.get(global_idx, None)

    def local_to_global(self, local_idx: int) -> int:
        """Convert local node index to global index."""
        return self.global_ids[local_idx].item()

    def contains_node(self, global_idx: int) -> bool:
        """Check if a global node belongs to this client."""
        return global_idx in self.local_map

    @property
    def num_nodes(self) -> int:
        return len(self.global_ids)

    @property
    def num_cross_edges(self) -> int:
        return self.cross_edges.size(1) if self.cross_edges.numel() > 0 else 0


def build_client_subgraph(
    full_data: Data,
    node_indices: torch.Tensor,
    full_edge_index: torch.Tensor,
    client_id: int
) -> SubgraphResult:
    """
    Extract a client subgraph from the full graph.

    Only intra-client edges (both endpoints in this client) are kept.
    Cross-client edges are recorded separately for analysis.

    Args:
        full_data: Full graph Data object
        node_indices: Global node IDs assigned to this client [n_k]
        full_edge_index: Full graph edge_index [2, E]
        client_id: Client identifier

    Returns:
        SubgraphResult with subgraph data and index mappings
    """
    device = full_data.x.device
    node_indices = node_indices.to(device)

    # Sort node indices for consistent mapping
    node_indices_sorted, sort_order = torch.sort(node_indices)

    # Build global -> local mapping
    local_map = {}
    for local_idx, global_idx in enumerate(node_indices_sorted.tolist()):
        local_map[global_idx] = local_idx

    # Filter intra-client edges (both endpoints in this client)
    intra_mask = filter_edges_by_nodes_vectorized(
        full_edge_index, node_indices_sorted, device
    )

    # Remap to local indices
    local_edge_index = remap_edge_indices(
        full_edge_index, intra_mask, node_indices_sorted, device
    )

    # Identify cross-client edges: exactly one endpoint in this client
    row, col = full_edge_index
    num_total_nodes = max(row.max().item(), col.max().item()) + 1
    node_mask = torch.zeros(num_total_nodes, dtype=torch.bool, device=device)
    node_mask[node_indices_sorted] = True

    row_in = node_mask[row]
    col_in = node_mask[col]
    cross_mask = row_in ^ col_in  # XOR: exactly one endpoint in client
    cross_edges = full_edge_index[:, cross_mask]

    # Build subgraph Data
    sub_data = Data(
        x=full_data.x[node_indices_sorted],
        y=full_data.y[node_indices_sorted],
        edge_index=local_edge_index,
        num_nodes=len(node_indices_sorted),
    )

    # Transfer masks if available
    if hasattr(full_data, 'train_mask') and full_data.train_mask is not None:
        sub_data.train_mask = full_data.train_mask[node_indices_sorted]
    if hasattr(full_data, 'val_mask') and full_data.val_mask is not None:
        sub_data.val_mask = full_data.val_mask[node_indices_sorted]
    if hasattr(full_data, 'test_mask') and full_data.test_mask is not None:
        sub_data.test_mask = full_data.test_mask[node_indices_sorted]

    return SubgraphResult(
        data=sub_data,
        global_ids=node_indices_sorted,
        local_map=local_map,
        client_id=client_id,
        cross_edges=cross_edges,
    )


def build_client_subgraph_with_neighbors(
    full_data: Data,
    node_indices: torch.Tensor,
    full_edge_index: torch.Tensor,
    client_id: int,
) -> SubgraphResult:
    """
    Build a client subgraph that includes cross-client neighbor features.

    Simulates FedSage-style neighbor generation: for each cross-client edge,
    the remote node's features are included as an "extended" node in the
    subgraph, and the cross-client edge is added to the local edge_index.

    This enables message passing across client boundaries, unlike the
    default build_client_subgraph which drops cross-client edges.

    Args:
        full_data: Full graph Data object
        node_indices: Global node IDs assigned to this client [n_k]
        full_edge_index: Full graph edge_index [2, E]
        client_id: Client identifier

    Returns:
        SubgraphResult with extended subgraph including neighbor nodes
    """
    device = full_data.x.device
    node_indices = node_indices.to(device)
    node_indices_sorted, _ = torch.sort(node_indices)

    client_set = set(node_indices_sorted.tolist())

    # Find cross-client neighbor nodes
    row, col = full_edge_index
    num_total = max(row.max().item(), col.max().item()) + 1
    node_mask = torch.zeros(num_total, dtype=torch.bool, device=device)
    node_mask[node_indices_sorted] = True

    row_in = node_mask[row]
    col_in = node_mask[col]

    # Cross edges: exactly one endpoint in client
    cross_mask = row_in ^ col_in
    cross_edge_index = full_edge_index[:, cross_mask]

    # Collect remote neighbor node IDs
    remote_nodes = set()
    for i in range(cross_edge_index.size(1)):
        r, c = cross_edge_index[0, i].item(), cross_edge_index[1, i].item()
        if r not in client_set:
            remote_nodes.add(r)
        if c not in client_set:
            remote_nodes.add(c)

    remote_nodes = sorted(remote_nodes)

    # Extended node set: client nodes + remote neighbors
    extended_ids = torch.cat([
        node_indices_sorted,
        torch.tensor(remote_nodes, dtype=torch.long, device=device),
    ])

    # Build local map for extended set
    local_map = {}
    for local_idx, global_idx in enumerate(extended_ids.tolist()):
        local_map[global_idx] = local_idx

    # Filter edges: keep edges where BOTH endpoints are in extended set
    ext_mask = torch.zeros(num_total, dtype=torch.bool, device=device)
    ext_mask[extended_ids] = True
    both_in = ext_mask[row] & ext_mask[col]
    kept_edges = full_edge_index[:, both_in]

    # Remap to local indices
    global_to_local_map = torch.full((num_total,), -1, dtype=torch.long, device=device)
    global_to_local_map[extended_ids] = torch.arange(len(extended_ids), device=device)
    local_edge_index = torch.stack([
        global_to_local_map[kept_edges[0]],
        global_to_local_map[kept_edges[1]],
    ], dim=0)

    # Build Data with extended features
    sub_data = Data(
        x=full_data.x[extended_ids],
        y=full_data.y[extended_ids],
        edge_index=local_edge_index,
        num_nodes=len(extended_ids),
    )

    # Masks: only client-owned nodes are trainable
    n_client = len(node_indices_sorted)
    n_extended = len(extended_ids)
    train_mask = torch.zeros(n_extended, dtype=torch.bool, device=device)
    train_mask[:n_client] = True
    if hasattr(full_data, 'train_mask') and full_data.train_mask is not None:
        orig_mask = full_data.train_mask[node_indices_sorted]
        train_mask[:n_client] = orig_mask
    sub_data.train_mask = train_mask

    # Record original cross edges for analysis
    cross_edges_record = full_edge_index[:, cross_mask]

    return SubgraphResult(
        data=sub_data,
        global_ids=extended_ids,
        local_map=local_map,
        client_id=client_id,
        cross_edges=cross_edges_record,
    )


def filter_nodes_in_client(
    global_indices: list,
    subgraph: SubgraphResult
) -> tuple:
    """
    Filter a list of global node indices to those present in a client's subgraph.

    Returns both the filtered global indices and their local counterparts.

    Args:
        global_indices: List of global node indices
        subgraph: Client's SubgraphResult

    Returns:
        Tuple of (filtered_global_indices, corresponding_local_indices)
    """
    filtered_global = []
    filtered_local = []
    for gid in global_indices:
        lid = subgraph.global_to_local(gid)
        if lid is not None:
            filtered_global.append(gid)
            filtered_local.append(lid)
    return filtered_global, filtered_local
