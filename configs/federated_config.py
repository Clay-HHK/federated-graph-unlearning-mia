"""
Configuration for federated graph unlearning privacy audit experiments.

Extends the centralized Config with federated-specific parameters:
client counts, partition methods, data distributions, re-aggregation rounds,
audit levels, and threat models.
"""

from dataclasses import dataclass, field
from typing import List

from .default import Config


@dataclass
class FederatedConfig(Config):
    """Federated experiment configuration extending centralized Config.

    Inherits all centralized parameters (seed, epochs, hidden_channels, etc.)
    and adds federated-specific settings.
    """

    # ==================== Federated Training ====================
    num_clients: int = 5
    num_rounds: int = 50
    local_epochs: int = 5
    partition_method: str = 'metis'
    data_distribution: str = 'iid'
    label_skew_alpha: float = 0.5

    # ==================== Federated Unlearning ====================
    reaggregate_rounds: int = 5
    fed_unlearn_methods: List[str] = field(default_factory=lambda: [
        'FedRetrain', 'FedGNNDelete', 'FedGraphEraser',
    ])
    unlearn_granularity: str = 'node'  # 'node' or 'client'

    # ==================== Attack / Audit ====================
    audit_levels: List[str] = field(default_factory=lambda: [
        'global', 'local', 'cross_client',
    ])

    # ==================== Experiment Matrix ====================
    client_counts: List[int] = field(default_factory=lambda: [3, 5, 10])
    partition_methods: List[str] = field(default_factory=lambda: ['metis', 'random'])
    data_distributions: List[str] = field(default_factory=lambda: ['iid', 'label_skew'])
    reaggregate_options: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # ==================== Federated Output ====================
    fed_results_dir: str = 'results/federated'
    fed_tables_dir: str = 'results/federated/tables'
    fed_figures_dir: str = 'results/federated/figures'

    def __post_init__(self):
        """Validate federated-specific parameters."""
        super().__post_init__()
        assert self.num_clients > 0, "num_clients must be positive"
        assert self.num_rounds > 0, "num_rounds must be positive"
        assert self.local_epochs > 0, "local_epochs must be positive"
        assert self.reaggregate_rounds >= 0, "reaggregate_rounds must be non-negative"
        assert self.partition_method in ('metis', 'random', 'community'), \
            f"Unknown partition method: {self.partition_method}"
        assert self.data_distribution in ('iid', 'label_skew'), \
            f"Unknown distribution: {self.data_distribution}"
        assert self.unlearn_granularity in ('node', 'client'), \
            f"Unknown granularity: {self.unlearn_granularity}"

    def ensure_fed_dirs(self):
        """Create federated output directories."""
        import os
        self.ensure_dirs()
        for d in [self.fed_results_dir, self.fed_tables_dir, self.fed_figures_dir]:
            os.makedirs(d, exist_ok=True)

    def summary(self) -> str:
        """Extended summary including federated settings."""
        base = super().summary()
        fed_lines = [
            "",
            "-" * 60,
            "Federated Settings:",
            f"  Num Clients: {self.num_clients}",
            f"  Num Rounds: {self.num_rounds}",
            f"  Local Epochs: {self.local_epochs}",
            f"  Partition: {self.partition_method}",
            f"  Distribution: {self.data_distribution}",
            f"  Re-agg Rounds: {self.reaggregate_rounds}",
            f"  Unlearning Methods: {self.fed_unlearn_methods}",
            f"  Granularity: {self.unlearn_granularity}",
            f"  Audit Levels: {self.audit_levels}",
            "=" * 60,
        ]
        return base + "\n".join(fed_lines)


# ==================== Preset Configurations ====================

def get_fed_pilot_config() -> FederatedConfig:
    """Quick pilot: 1 dataset, 3 clients, 10 trials, ~5min."""
    return FederatedConfig(
        num_trials=10,
        datasets=['Cora'],
        gnn_models=['GCN-2L'],
        num_clients=3,
        num_rounds=30,
        local_epochs=3,
        partition_method='metis',
        data_distribution='iid',
        reaggregate_rounds=5,
        fed_unlearn_methods=['FedRetrain', 'FedGNNDelete'],
        epochs=100,
    )


def get_fed_standard_config() -> FederatedConfig:
    """Standard federated experiment: 5 datasets, full matrix."""
    return FederatedConfig(
        num_trials=50,
        datasets=['Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel'],
        gnn_models=['GCN-2L'],
    )


def get_fed_debug_config() -> FederatedConfig:
    """Minimal debug config for development."""
    return FederatedConfig(
        num_trials=2,
        datasets=['Cora'],
        gnn_models=['GCN-2L'],
        num_clients=3,
        num_rounds=10,
        local_epochs=2,
        reaggregate_rounds=2,
        fed_unlearn_methods=['FedRetrain'],
        epochs=50,
    )
