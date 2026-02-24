"""
Unified Configuration for Graph Unlearning Privacy Audit Experiments.

Paper: Confidence Illusion vs L2 Geometric Truth in Graph Unlearning

This module provides a centralized configuration system. All experiment scripts
should import from here to ensure consistency.

Usage:
    from configs.default import Config

    # Use default config
    config = Config()

    # Or override specific settings
    config = Config(num_trials=50, epochs=100)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


# ==================== Main Configuration Class ====================

@dataclass
class Config:
    """
    Unified configuration for all experiments.

    All default values are defined here. Override by passing kwargs to __init__.

    Example:
        >>> config = Config()  # Use all defaults
        >>> config = Config(num_trials=50, seed=123)  # Override specific values
    """

    # ==================== Reproducibility ====================
    seed: int = 42

    # ==================== Experiment Settings ====================
    num_trials: int = 100

    # Datasets to evaluate
    datasets: List[str] = field(default_factory=lambda: [
        'Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel'
    ])

    # GNN architectures
    gnn_models: List[str] = field(default_factory=lambda: [
        'GCN-2L', 'GCN-3L', 'GAT', 'GraphSAGE'
    ])

    # Unlearning methods
    unlearn_methods: List[str] = field(default_factory=lambda: [
        'Retrain',           # Gold standard (full retraining)
        'GNNDelete',         # Gradient-based unlearning
        'GIF',               # Graph Influence Function
        'GraphEditor',       # Entropy maximization
        'GraphEraser-BEKM',  # Shard-based with balanced K-means
        'GraphEraser-BLPA',  # Shard-based with label propagation
    ])

    # ==================== Model Hyperparameters ====================
    hidden_channels: int = 16
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200

    # ==================== Unlearning Parameters ====================
    unlearn_steps: int = 20        # Steps for gradient-based methods
    num_shards: int = 5            # Number of shards for SISA-based methods
    balance_ratio: float = 1.1     # Balance constraint for partitioning
    blpa_max_iters: int = 10       # Max iterations for BLPA
    blpa_convergence: float = 0.01 # Convergence threshold for BLPA

    # ==================== Attack Evaluation ====================
    # AUC thresholds for vulnerability assessment
    auc_highly_vulnerable: float = 0.90
    auc_moderately_vulnerable: float = 0.70

    # Signal-to-noise ratio thresholds
    snr_strong_signal: float = 3.0
    snr_weak_signal: float = 1.5

    # Number of control nodes for AUC computation
    min_control_nodes: int = 5

    # ==================== Output Settings ====================
    results_dir: str = 'results'
    tables_dir: str = 'results/tables'
    figures_dir: str = 'results/figures'
    logs_dir: str = 'results/logs'
    main_results_file: str = 'main_experiment_results.csv'

    # ==================== Data Paths ====================
    data_root: str = '/tmp'

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.seed >= 0, "Seed must be non-negative"
        assert self.num_trials > 0, "num_trials must be positive"
        assert 0 < self.dropout < 1, "Dropout must be between 0 and 1"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.epochs > 0, "Epochs must be positive"
        assert self.num_shards > 0, "num_shards must be positive"
        assert self.balance_ratio >= 1.0, "balance_ratio must be >= 1.0"

    def get_device(self) -> torch.device:
        """Get the appropriate device (CUDA if available)."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_dataset_path(self, dataset_name: str) -> str:
        """Get the cache path for a dataset."""
        return os.path.join(self.data_root, dataset_name)

    def get_results_path(self, filename: str = None) -> str:
        """Get full path for results file."""
        if filename is None:
            filename = self.main_results_file
        return os.path.join(self.tables_dir, filename)

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        for dir_path in [self.results_dir, self.tables_dir,
                         self.figures_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'seed': self.seed,
            'num_trials': self.num_trials,
            'datasets': self.datasets,
            'gnn_models': self.gnn_models,
            'unlearn_methods': self.unlearn_methods,
            'hidden_channels': self.hidden_channels,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'unlearn_steps': self.unlearn_steps,
            'num_shards': self.num_shards,
            'balance_ratio': self.balance_ratio,
        }

    def summary(self) -> str:
        """Return a formatted summary of the configuration."""
        lines = [
            "=" * 60,
            "Experiment Configuration",
            "=" * 60,
            f"Seed: {self.seed}",
            f"Num Trials: {self.num_trials}",
            f"Datasets: {self.datasets}",
            f"GNN Models: {self.gnn_models}",
            f"Unlearning Methods: {self.unlearn_methods}",
            "-" * 60,
            "Model Hyperparameters:",
            f"  Hidden Channels: {self.hidden_channels}",
            f"  Dropout: {self.dropout}",
            f"  Learning Rate: {self.learning_rate}",
            f"  Weight Decay: {self.weight_decay}",
            f"  Epochs: {self.epochs}",
            "-" * 60,
            "Unlearning Parameters:",
            f"  Unlearn Steps: {self.unlearn_steps}",
            f"  Num Shards: {self.num_shards}",
            f"  Balance Ratio: {self.balance_ratio}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ==================== Preset Configurations ====================

def get_quick_config() -> Config:
    """Quick configuration for testing (reduced trials and datasets)."""
    return Config(
        num_trials=10,
        datasets=['Cora', 'CiteSeer'],
        gnn_models=['GCN-2L'],
        epochs=100
    )


def get_standard_config() -> Config:
    """Standard configuration for full experiments."""
    return Config()


def get_comprehensive_config() -> Config:
    """Comprehensive configuration with all datasets."""
    return Config(
        datasets=[
            'Cora', 'CiteSeer', 'PubMed',           # Homophilic
            'Chameleon', 'Squirrel',                 # Heterophilic
            'Amazon-Computers', 'Amazon-Photo',      # Co-purchase
            'Coauthor-CS',                           # Co-authorship
        ],
        num_trials=100
    )


def get_debug_config() -> Config:
    """Minimal configuration for debugging."""
    return Config(
        num_trials=3,
        datasets=['Cora'],
        gnn_models=['GCN-2L'],
        unlearn_methods=['Retrain', 'GNNDelete'],
        epochs=50
    )


# ==================== Dataset Metadata ====================

DATASET_INFO = {
    'Cora': {
        'type': 'citation',
        'homophily': 'high',
        'expected_h': 0.81,
        'nodes': 2708,
        'edges': 10556,
        'features': 1433,
        'classes': 7
    },
    'CiteSeer': {
        'type': 'citation',
        'homophily': 'high',
        'expected_h': 0.74,
        'nodes': 3327,
        'edges': 9104,
        'features': 3703,
        'classes': 6
    },
    'PubMed': {
        'type': 'citation',
        'homophily': 'high',
        'expected_h': 0.80,
        'nodes': 19717,
        'edges': 88648,
        'features': 500,
        'classes': 3
    },
    'Chameleon': {
        'type': 'wikipedia',
        'homophily': 'low',
        'expected_h': 0.23,
        'nodes': 2277,
        'edges': 36101,
        'features': 2325,
        'classes': 5
    },
    'Squirrel': {
        'type': 'wikipedia',
        'homophily': 'low',
        'expected_h': 0.22,
        'nodes': 5201,
        'edges': 217073,
        'features': 2089,
        'classes': 5
    },
    'Amazon-Computers': {
        'type': 'copurchase',
        'homophily': 'medium',
        'expected_h': 0.78,
        'nodes': 13752,
        'edges': 491722,
        'features': 767,
        'classes': 10
    },
    'Amazon-Photo': {
        'type': 'copurchase',
        'homophily': 'medium',
        'expected_h': 0.83,
        'nodes': 7650,
        'edges': 238162,
        'features': 745,
        'classes': 8
    },
    'Coauthor-CS': {
        'type': 'coauthor',
        'homophily': 'high',
        'expected_h': 0.81,
        'nodes': 18333,
        'edges': 163788,
        'features': 6805,
        'classes': 15
    },
}


# ==================== Legacy Compatibility ====================
# These classes are kept for backward compatibility with old code

class ExperimentConfig:
    """Legacy class - use Config instead."""
    SEED = 42
    NUM_TRIALS = 100
    DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel']
    GNN_MODELS = ['GCN-2L', 'GCN-3L', 'GAT', 'GraphSAGE']
    UNLEARN_METHODS = [
        'Retrain', 'GNNDelete', 'GIF', 'GraphEditor',
        'GraphEraser-BEKM', 'GraphEraser-BLPA'
    ]


class ModelConfig:
    """Legacy class - use Config instead."""
    HIDDEN_CHANNELS = 16
    DROPOUT = 0.5
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 5e-4
    EPOCHS = 200


class UnlearnConfig:
    """Legacy class - use Config instead."""
    UNLEARN_STEPS = 20
    NUM_SHARDS = 5
    BALANCE_RATIO = 1.1


class OutputConfig:
    """Legacy class - use Config instead."""
    RESULTS_DIR = 'results'
    TABLES_DIR = 'results/tables'
    FIGURES_DIR = 'results/figures'
    LOGS_DIR = 'results/logs'
    MAIN_RESULTS_FILE = 'main_experiment_results.csv'
