"""
Graph unlearning methods for Hub-Ripple MIA experiments.

This module provides various unlearning algorithms:
- Baseline: Complete retraining (ground truth)
- GNNDelete: Gradient-based forgetting
- GIF: Graph Influence Function
- SISA: Sharded unlearning (BEKM, BLPA, Random)
- GraphEditor: Edge weight learning
- GraphEraser: Certified deletion

All methods follow a consistent interface for easy experimentation.
"""

from .baseline import unlearn_baseline
from .gnn_delete import unlearn_gnndelete
from .gif import unlearn_gif
from .sisa import (
    unlearn_sisa,
    partition_bekm,
    partition_blpa,
    partition_random,
    build_shard_data
)
from .graph_editor import (
    unlearn_graph_editor,
    unlearn_graph_editor_with_neighbors,
    unlearn_grapheditor  # Alias
)
from .graph_eraser import (
    unlearn_graph_eraser,
    unlearn_graph_eraser_ensemble,
    partition_graph_eraser,
    build_shard_subgraph
)

__all__ = [
    # Main unlearning methods
    'unlearn_baseline',
    'unlearn_gnndelete',
    'unlearn_gif',
    'unlearn_sisa',
    'unlearn_graph_editor',
    'unlearn_graph_eraser',

    # Aliases
    'unlearn_grapheditor',

    # SISA variants
    'partition_bekm',
    'partition_blpa',
    'partition_random',
    'build_shard_data',

    # GraphEditor variants
    'unlearn_graph_editor_with_neighbors',

    # GraphEraser variants
    'unlearn_graph_eraser_ensemble',
    'partition_graph_eraser',
    'build_shard_subgraph',
]


# ==================== Convenience Functions ====================

def get_unlearning_method(method_name: str):
    """
    Get unlearning method by name.

    Args:
        method_name: Method name (case-insensitive)
            - 'baseline' or 'retrain'
            - 'gnndelete' or 'delete'
            - 'gif' or 'influence'
            - 'sisa' or 'sisa-bekm'
            - 'sisa-blpa'
            - 'sisa-random'
            - 'grapheditor' or 'editor'
            - 'grapheraser' or 'eraser'

    Returns:
        Unlearning function

    Example:
        >>> unlearn_func = get_unlearning_method('baseline')
        >>> model_new = unlearn_func(model, data, target_idx, ...)
    """
    method_name = method_name.lower().replace('_', '').replace('-', '')

    methods = {
        'baseline': unlearn_baseline,
        'retrain': unlearn_baseline,
        'gnndelete': unlearn_gnndelete,
        'delete': unlearn_gnndelete,
        'gif': unlearn_gif,
        'influence': unlearn_gif,
        'sisa': unlearn_sisa,
        'sisabekm': unlearn_sisa,
        'sisablpa': lambda *args, **kwargs: unlearn_sisa(*args, partition_strategy='blpa', **kwargs),
        'sisarandom': lambda *args, **kwargs: unlearn_sisa(*args, partition_strategy='random', **kwargs),
        'grapheditor': unlearn_graph_editor,
        'editor': unlearn_graph_editor,
        'grapheraser': unlearn_graph_eraser,
        'eraser': unlearn_graph_eraser,
    }

    if method_name not in methods:
        raise ValueError(
            f"Unknown method: {method_name}. "
            f"Available: {list(set(methods.keys()))}"
        )

    return methods[method_name]


def list_available_methods():
    """
    List all available unlearning methods.

    Returns:
        List of method names

    Example:
        >>> methods = list_available_methods()
        >>> print(methods)
    """
    return [
        'Baseline',
        'GNNDelete',
        'GIF',
        'SISA-BEKM',
        'SISA-BLPA',
        'SISA-Random',
        'GraphEditor',
        'GraphEraser'
    ]


def get_method_description(method_name: str) -> str:
    """
    Get description of an unlearning method.

    Args:
        method_name: Method name

    Returns:
        Description string

    Example:
        >>> desc = get_method_description('Baseline')
        >>> print(desc)
    """
    descriptions = {
        'Baseline': 'Complete retraining from scratch (ground truth unlearning)',
        'GNNDelete': 'Gradient-based forgetting with neighbor retention',
        'GIF': 'Graph Influence Function approximation',
        'SISA-BEKM': 'SISA with Balanced K-Means partitioning',
        'SISA-BLPA': 'SISA with Balanced Label Propagation partitioning',
        'SISA-Random': 'SISA with Random partitioning',
        'GraphEditor': 'Edge weight learning for soft deletion',
        'GraphEraser': 'Certified deletion with optimized partitioning'
    }

    method_name = method_name.replace('_', '-')
    if method_name not in descriptions:
        return "Unknown method"

    return descriptions[method_name]
