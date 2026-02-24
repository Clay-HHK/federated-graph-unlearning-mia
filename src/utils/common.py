"""
Common utilities and constants used across experiments.
"""

import torch
import numpy as np
import random


# ==================== Default Hyperparameters ====================

# Model architecture
DEFAULT_HIDDEN_CHANNELS = 16
DEFAULT_DROPOUT = 0.5

# Training
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_EPOCHS = 200

# Unlearning methods
DEFAULT_UNLEARN_STEPS = 20  # For gradient-based methods
DEFAULT_NUM_SHARDS = 5      # For SISA-based methods
DEFAULT_BALANCE_RATIO = 1.1  # For BEKM partitioning

# Experiments
DEFAULT_SEED = 42
DEFAULT_NUM_TRIALS = 20


# ==================== Reproducibility ====================

def set_seed(seed: int = DEFAULT_SEED):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed (default: 42)

    Example:
        >>> set_seed(42)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get available device (CUDA if available, else CPU).

    Returns:
        torch.device

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== Dataset Paths ====================

def get_dataset_path(dataset_name: str) -> str:
    """
    Get standard dataset cache path.

    Args:
        dataset_name: Name of dataset (e.g., 'Cora', 'CiteSeer')

    Returns:
        Path to dataset cache directory

    Example:
        >>> path = get_dataset_path('Cora')
        >>> dataset = Planetoid(root=path, name='Cora')
    """
    return f'/tmp/{dataset_name}'


# ==================== Hub-Ripple Constants ====================

# Attack success thresholds
AUC_HIGHLY_VULNERABLE = 0.90
AUC_MODERATELY_VULNERABLE = 0.70

# Signal-to-noise ratio thresholds
SNR_STRONG_SIGNAL = 3.0
SNR_WEAK_SIGNAL = 1.5


def interpret_auc(auc: float) -> str:
    """
    Interpret AUC score for Hub-Ripple MIA.

    Args:
        auc: AUC score [0, 1]

    Returns:
        Vulnerability interpretation

    Example:
        >>> print(interpret_auc(0.95))
        'HIGHLY VULNERABLE to Hub-Ripple MIA'
    """
    if auc >= AUC_HIGHLY_VULNERABLE:
        return "HIGHLY VULNERABLE to Hub-Ripple MIA"
    elif auc >= AUC_MODERATELY_VULNERABLE:
        return "MODERATELY VULNERABLE to Hub-Ripple MIA"
    else:
        return "RESISTANT to Hub-Ripple MIA"


def interpret_snr(snr: float) -> str:
    """
    Interpret signal-to-noise ratio.

    Args:
        snr: Signal-to-noise ratio

    Returns:
        Signal strength interpretation

    Example:
        >>> print(interpret_snr(5.2))
        'Strong signal (5.2x)'
    """
    if snr >= SNR_STRONG_SIGNAL:
        return f"Strong signal ({snr:.1f}x)"
    elif snr >= SNR_WEAK_SIGNAL:
        return f"Moderate signal ({snr:.1f}x)"
    else:
        return f"Weak signal ({snr:.1f}x)"


# ==================== Experiment Metadata ====================

# Standard datasets for multi-dataset experiments
STANDARD_DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'Chameleon', 'Squirrel']

# Homophilic datasets
HOMOPHILIC_DATASETS = ['Cora', 'CiteSeer', 'PubMed']

# Heterophilic datasets
HETEROPHILIC_DATASETS = ['Chameleon', 'Squirrel']

# All unlearning methods
ALL_UNLEARNING_METHODS = ['Baseline', 'GNNDelete', 'GIF', 'SISA-BEKM', 'GraphEditor']

# Partitioning strategies
PARTITIONING_STRATEGIES = ['Random', 'BEKM', 'BLPA']


# ==================== Statistical Validation ====================

# Minimum samples for reliable AUC computation
MIN_AUC_SAMPLES = 5

# Confidence interval parameters
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_ITERATIONS = 1000


def validate_sample_size(n_positive: int, n_negative: int, min_samples: int = MIN_AUC_SAMPLES) -> bool:
    """
    Check if sample size is sufficient for reliable AUC computation.

    Args:
        n_positive: Number of positive samples
        n_negative: Number of negative samples
        min_samples: Minimum required samples per class

    Returns:
        True if sample size is sufficient

    Example:
        >>> if not validate_sample_size(len(hubs), len(controls)):
        ...     print("Warning: AUC may be unreliable")
    """
    return n_positive >= min_samples and n_negative >= min_samples


def compute_confidence_interval_bootstrap(
    values: list,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    n_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS
) -> tuple:
    """
    Compute confidence interval using bootstrap resampling.

    Args:
        values: List of observed values
        confidence: Confidence level (default: 0.95)
        n_iterations: Number of bootstrap iterations

    Returns:
        Tuple of (mean, lower_bound, upper_bound)

    Example:
        >>> mean, lower, upper = compute_confidence_interval_bootstrap(auc_scores)
        >>> print(f"AUC: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    bootstrap_means = []
    for _ in range(n_iterations):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    mean = np.mean(values)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(mean), float(lower), float(upper)
