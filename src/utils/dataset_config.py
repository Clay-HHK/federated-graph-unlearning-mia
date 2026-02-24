"""
Dataset configuration and metadata for multi-dataset experiments.

This module provides comprehensive dataset support for Hub-Ripple MIA,
including homophilic (同质图) and heterophilic (异质图) graphs.
"""

import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Amazon, Coauthor, WebKB, Actor
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import numpy as np


# ==================== Dataset Metadata ====================

DATASET_METADATA = {
    # Citation Networks (引文网络) - 高同质性
    'Cora': {
        'type': 'citation',
        'homophily': 'high',
        'expected_h': 0.81,
        'nodes': 2708,
        'edges': 10556,
        'features': 1433,
        'classes': 7,
        'source': 'Planetoid',
        'description': 'Citation network of machine learning papers'
    },
    'CiteSeer': {
        'type': 'citation',
        'homophily': 'high',
        'expected_h': 0.74,
        'nodes': 3327,
        'edges': 9104,
        'features': 3703,
        'classes': 6,
        'source': 'Planetoid',
        'description': 'Citation network of scientific publications'
    },
    'PubMed': {
        'type': 'citation',
        'homophily': 'high',
        'expected_h': 0.80,
        'nodes': 19717,
        'edges': 88648,
        'features': 500,
        'classes': 3,
        'source': 'Planetoid',
        'description': 'Citation network of diabetes research papers'
    },

    # Wikipedia Networks (维基百科网络) - 低同质性/异质性
    'Chameleon': {
        'type': 'wikipedia',
        'homophily': 'low',
        'expected_h': 0.23,
        'nodes': 2277,
        'edges': 36101,
        'features': 2325,
        'classes': 5,
        'source': 'WikipediaNetwork',
        'description': 'Wikipedia page-page network about chameleons'
    },
    'Squirrel': {
        'type': 'wikipedia',
        'homophily': 'low',
        'expected_h': 0.22,
        'nodes': 5201,
        'edges': 217073,
        'features': 2089,
        'classes': 5,
        'source': 'WikipediaNetwork',
        'description': 'Wikipedia page-page network about squirrels'
    },

    # Co-purchase Networks (共同购买网络) - 中等同质性
    'Amazon-Computers': {
        'type': 'copurchase',
        'homophily': 'medium',
        'expected_h': 0.78,
        'nodes': 13752,
        'edges': 491722,
        'features': 767,
        'classes': 10,
        'source': 'Amazon',
        'description': 'Amazon co-purchase network (Computers category)'
    },
    'Amazon-Photo': {
        'type': 'copurchase',
        'homophily': 'medium',
        'expected_h': 0.83,
        'nodes': 7650,
        'edges': 238162,
        'features': 745,
        'classes': 8,
        'source': 'Amazon',
        'description': 'Amazon co-purchase network (Photo category)'
    },

    # Co-authorship Networks (共同作者网络) - 高同质性
    'Coauthor-CS': {
        'type': 'coauthor',
        'homophily': 'high',
        'expected_h': 0.81,
        'nodes': 18333,
        'edges': 163788,
        'features': 6805,
        'classes': 15,
        'source': 'Coauthor',
        'description': 'Co-authorship network (Computer Science)'
    },
    'Coauthor-Physics': {
        'type': 'coauthor',
        'homophily': 'high',
        'expected_h': 0.93,
        'nodes': 34493,
        'edges': 495924,
        'features': 8415,
        'classes': 5,
        'source': 'Coauthor',
        'description': 'Co-authorship network (Physics)'
    },

    # WebKB Networks (大学网页网络) - 中等同质性
    'Cornell': {
        'type': 'webkb',
        'homophily': 'low',
        'expected_h': 0.30,
        'nodes': 183,
        'edges': 298,
        'features': 1703,
        'classes': 5,
        'source': 'WebKB',
        'description': 'Cornell university webpage network'
    },
    'Texas': {
        'type': 'webkb',
        'homophily': 'low',
        'expected_h': 0.11,
        'nodes': 183,
        'edges': 325,
        'features': 1703,
        'classes': 5,
        'source': 'WebKB',
        'description': 'Texas university webpage network'
    },
    'Wisconsin': {
        'type': 'webkb',
        'homophily': 'low',
        'expected_h': 0.21,
        'nodes': 251,
        'edges': 515,
        'features': 1703,
        'classes': 5,
        'source': 'WebKB',
        'description': 'Wisconsin university webpage network'
    },

    # Actor Network (演员网络) - 低同质性
    'Actor': {
        'type': 'actor',
        'homophily': 'low',
        'expected_h': 0.22,
        'nodes': 7600,
        'edges': 33544,
        'features': 931,
        'classes': 5,
        'source': 'Actor',
        'description': 'Actor co-occurrence network'
    }
}


# ==================== Predefined Dataset Configurations ====================

# Standard configuration (5 datasets: 3 homophilic + 2 heterophilic)
STANDARD_DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel']

# Homophilic graphs (同质图)
HOMOPHILIC_DATASETS = [
    'Cora', 'CiteSeer', 'PubMed',           # Citation networks
    'Amazon-Computers', 'Amazon-Photo',      # Co-purchase
    'Coauthor-CS', 'Coauthor-Physics'       # Co-authorship
]

# Heterophilic graphs (异质图)
HETEROPHILIC_DATASETS = [
    'Chameleon', 'Squirrel',                # Wikipedia
    'Cornell', 'Texas', 'Wisconsin',         # WebKB
    'Actor'                                  # Actor network
]

# Small graphs (<5K nodes) - for quick experiments
SMALL_DATASETS = [
    'Cora', 'CiteSeer', 'Chameleon', 'Squirrel',
    'Cornell', 'Texas', 'Wisconsin'
]

# Medium graphs (5K-20K nodes)
MEDIUM_DATASETS = [
    'PubMed', 'Amazon-Photo', 'Amazon-Computers',
    'Actor', 'Coauthor-CS'
]

# Large graphs (>20K nodes)
LARGE_DATASETS = [
    'Coauthor-Physics'
]

# Comprehensive evaluation (all datasets)
ALL_DATASETS = list(DATASET_METADATA.keys())


# ==================== Dataset Loading ====================

def load_dataset_unified(
    name: str,
    device: torch.device,
    data_root: str = '/tmp'
) -> Tuple[Data, Dict]:
    """
    Unified dataset loader with automatic configuration.

    Args:
        name: Dataset name (e.g., 'Cora', 'Chameleon')
        device: Device to load data onto
        data_root: Root directory for dataset cache

    Returns:
        Tuple of (data, metadata_dict)

    Example:
        >>> device = torch.device('cuda')
        >>> data, meta = load_dataset_unified('Cora', device)
        >>> print(f"Homophily: {meta['homophily']:.3f}")
    """
    if name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_METADATA.keys())}")

    meta = DATASET_METADATA[name].copy()
    source = meta['source']

    # Load dataset based on source
    if source == 'Planetoid':
        dataset = Planetoid(root=f'{data_root}/{name}', name=name)
    elif source == 'WikipediaNetwork':
        dataset = WikipediaNetwork(root=f'{data_root}/{name}', name=name.lower())
    elif source == 'Amazon':
        category = name.split('-')[1]
        dataset = Amazon(root=f'{data_root}/{name}', name=category)
    elif source == 'Coauthor':
        field = name.split('-')[1]
        dataset = Coauthor(root=f'{data_root}/{name}', name=field)
    elif source == 'WebKB':
        dataset = WebKB(root=f'{data_root}/{name}', name=name)
    elif source == 'Actor':
        dataset = Actor(root=f'{data_root}/{name}')
    else:
        raise ValueError(f"Unknown dataset source: {source}")

    data = dataset[0].to(device)

    # Fix multi-dimensional masks (for Wikipedia, WebKB, Actor)
    if source in ['WikipediaNetwork', 'WebKB', 'Actor']:
        if hasattr(data, 'train_mask') and data.train_mask.dim() > 1:
            data.train_mask = data.train_mask[:, 0]
        if hasattr(data, 'val_mask') and data.val_mask.dim() > 1:
            data.val_mask = data.val_mask[:, 0]
        if hasattr(data, 'test_mask') and data.test_mask.dim() > 1:
            data.test_mask = data.test_mask[:, 0]

    # Ensure train_mask exists (some datasets may not have it)
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        # Create default 60/20/20 split
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[perm[:train_size]] = True
        data.val_mask[perm[train_size:train_size + val_size]] = True
        data.test_mask[perm[train_size + val_size:]] = True

    # Compute actual homophily
    homophily = compute_edge_homophily(data)
    meta['homophily'] = homophily
    meta['num_nodes'] = data.num_nodes
    meta['num_edges'] = data.edge_index.size(1)
    meta['num_features'] = data.num_features
    meta['num_classes'] = int(data.y.max().item()) + 1

    return data, meta


def compute_edge_homophily(data: Data) -> float:
    """
    Compute edge homophily ratio.

    Homophily = (# edges connecting same-label nodes) / (# total edges)

    Args:
        data: PyTorch Geometric data object

    Returns:
        Homophily ratio [0, 1]
    """
    row, col = data.edge_index
    same_label = (data.y[row] == data.y[col]).sum().item()
    total_edges = data.edge_index.size(1)
    homophily = same_label / total_edges if total_edges > 0 else 0.0
    return homophily


def get_dataset_info(name: str) -> Dict:
    """
    Get metadata for a dataset without loading it.

    Args:
        name: Dataset name

    Returns:
        Metadata dictionary

    Example:
        >>> info = get_dataset_info('Cora')
        >>> print(f"{info['description']}")
    """
    if name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_METADATA[name].copy()


def list_datasets_by_homophily(min_h: float = 0.0, max_h: float = 1.0) -> List[str]:
    """
    List datasets within a homophily range.

    Args:
        min_h: Minimum homophily
        max_h: Maximum homophily

    Returns:
        List of dataset names

    Example:
        >>> high_h = list_datasets_by_homophily(min_h=0.7)
        >>> low_h = list_datasets_by_homophily(max_h=0.3)
    """
    datasets = []
    for name, meta in DATASET_METADATA.items():
        h = meta['expected_h']
        if min_h <= h <= max_h:
            datasets.append(name)
    return datasets


def list_datasets_by_size(
    min_nodes: int = 0,
    max_nodes: int = float('inf')
) -> List[str]:
    """
    List datasets within a size range.

    Args:
        min_nodes: Minimum number of nodes
        max_nodes: Maximum number of nodes

    Returns:
        List of dataset names

    Example:
        >>> small = list_datasets_by_size(max_nodes=5000)
        >>> large = list_datasets_by_size(min_nodes=20000)
    """
    datasets = []
    for name, meta in DATASET_METADATA.items():
        n = meta['nodes']
        if min_nodes <= n <= max_nodes:
            datasets.append(name)
    return datasets


def print_dataset_summary():
    """
    Print comprehensive summary of all available datasets.

    Example:
        >>> print_dataset_summary()
    """
    print("\n" + "=" * 100)
    print("Available Datasets for Hub-Ripple MIA")
    print("=" * 100)

    # Group by homophily
    print("\n🔴 HIGH HOMOPHILY (Homophilic Graphs, 同质图):")
    print("-" * 100)
    print(f"{'Dataset':<20} {'Nodes':<8} {'Edges':<10} {'Classes':<8} {'H-ratio':<8} {'Description'}")
    print("-" * 100)

    for name in sorted(HOMOPHILIC_DATASETS):
        if name in DATASET_METADATA:
            m = DATASET_METADATA[name]
            print(f"{name:<20} {m['nodes']:<8} {m['edges']:<10} {m['classes']:<8} {m['expected_h']:<8.2f} {m['description'][:40]}")

    print("\n🔵 LOW HOMOPHILY (Heterophilic Graphs, 异质图):")
    print("-" * 100)
    print(f"{'Dataset':<20} {'Nodes':<8} {'Edges':<10} {'Classes':<8} {'H-ratio':<8} {'Description'}")
    print("-" * 100)

    for name in sorted(HETEROPHILIC_DATASETS):
        if name in DATASET_METADATA:
            m = DATASET_METADATA[name]
            print(f"{name:<20} {m['nodes']:<8} {m['edges']:<10} {m['classes']:<8} {m['expected_h']:<8.2f} {m['description'][:40]}")

    print("\n" + "=" * 100)
    print(f"Total: {len(DATASET_METADATA)} datasets")
    print(f"  Homophilic: {len(HOMOPHILIC_DATASETS)}")
    print(f"  Heterophilic: {len(HETEROPHILIC_DATASETS)}")
    print("=" * 100 + "\n")


# ==================== Experiment Configurations ====================

def get_experiment_config(
    config_name: str = 'standard'
) -> Dict[str, List[str]]:
    """
    Get predefined experiment configurations.

    Args:
        config_name: Configuration name
            - 'standard': 5 datasets (3 homo + 2 hetero)
            - 'quick': 4 small datasets
            - 'homophilic': All homophilic graphs
            - 'heterophilic': All heterophilic graphs
            - 'comprehensive': All available datasets
            - 'small': Small graphs only
            - 'medium': Medium graphs only

    Returns:
        Dictionary with 'datasets' key

    Example:
        >>> config = get_experiment_config('standard')
        >>> print(config['datasets'])
    """
    configs = {
        'standard': {
            'datasets': STANDARD_DATASETS,
            'description': 'Standard configuration (3 homophilic + 2 heterophilic)'
        },
        'quick': {
            'datasets': ['Cora', 'CiteSeer', 'Chameleon', 'Squirrel'],
            'description': 'Quick test (4 small datasets)'
        },
        'homophilic': {
            'datasets': HOMOPHILIC_DATASETS,
            'description': 'All homophilic graphs'
        },
        'heterophilic': {
            'datasets': HETEROPHILIC_DATASETS,
            'description': 'All heterophilic graphs'
        },
        'comprehensive': {
            'datasets': ALL_DATASETS,
            'description': 'All available datasets'
        },
        'small': {
            'datasets': SMALL_DATASETS,
            'description': 'Small graphs (<5K nodes)'
        },
        'medium': {
            'datasets': MEDIUM_DATASETS,
            'description': 'Medium graphs (5K-20K nodes)'
        },
        'citation': {
            'datasets': ['Cora', 'CiteSeer', 'PubMed'],
            'description': 'Citation networks only'
        },
        'wikipedia': {
            'datasets': ['Chameleon', 'Squirrel'],
            'description': 'Wikipedia networks only'
        }
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

    return configs[config_name]
