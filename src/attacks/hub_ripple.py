"""
Hub-Ripple Membership Inference Attack.

This module implements the core Hub-Ripple MIA attack for detecting
unlearning events in Graph Neural Networks.

Key Concept:
    When a target node is unlearned, its neighbors (hubs) show measurable
    embedding drift, while control nodes (non-neighbors) show minimal drift.
    The attack achieves high AUC (>0.90) by comparing hub vs control drift.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data

from ..models.training import get_embeddings
from ..utils.graph_utils import get_neighbors, select_control_node


def measure_embedding_drift(
    emb_before: torch.Tensor,
    emb_after: torch.Tensor,
    metric: str = 'l2'
) -> float:
    """
    Measure drift between two embeddings.

    Args:
        emb_before: Original embedding
        emb_after: New embedding after unlearning
        metric: Distance metric ('l2', 'l1', 'cosine')

    Returns:
        Drift value

    Example:
        >>> drift = measure_embedding_drift(hub_emb_orig, hub_emb_unlearned)
    """
    if metric == 'l2':
        drift = torch.norm(emb_before - emb_after, p=2).item()
    elif metric == 'l1':
        drift = torch.norm(emb_before - emb_after, p=1).item()
    elif metric == 'cosine':
        cos_sim = torch.nn.functional.cosine_similarity(
            emb_before.unsqueeze(0), emb_after.unsqueeze(0)
        ).item()
        drift = 1.0 - cos_sim
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return drift


def hub_ripple_single_trial(
    model_orig: torch.nn.Module,
    model_unlearned: torch.nn.Module,
    data_orig: Data,
    data_unlearned: Data,
    hub_idx: int,
    control_idx: int,
    metric: str = 'l2'
) -> Dict[str, float]:
    """
    Run single Hub-Ripple MIA trial.

    Args:
        model_orig: Original model (with target)
        model_unlearned: Unlearned model (without target)
        data_orig: Original graph data
        data_unlearned: Unlearned graph data
        hub_idx: Hub node index (target's neighbor)
        control_idx: Control node index (non-neighbor)
        metric: Distance metric for drift

    Returns:
        Dictionary with hub_drift, control_drift, ratio

    Example:
        >>> result = hub_ripple_single_trial(model_orig, model_unlearned,
        ...                                    data_orig, data_unlearned,
        ...                                    hub_idx, control_idx)
        >>> print(f"Hub drift: {result['hub_drift']:.3f}")
    """
    # Get embeddings from original model
    emb_orig = get_embeddings(model_orig, data_orig)
    hub_emb_orig = emb_orig[hub_idx]
    control_emb_orig = emb_orig[control_idx]

    # Get embeddings from unlearned model
    emb_unlearned = get_embeddings(model_unlearned, data_unlearned)
    hub_emb_unlearned = emb_unlearned[hub_idx]
    control_emb_unlearned = emb_unlearned[control_idx]

    # Measure drift
    hub_drift = measure_embedding_drift(hub_emb_orig, hub_emb_unlearned, metric)
    control_drift = measure_embedding_drift(control_emb_orig, control_emb_unlearned, metric)

    ratio = hub_drift / (control_drift + 1e-9)

    return {
        'hub_drift': hub_drift,
        'control_drift': control_drift,
        'ratio': ratio
    }


def hub_ripple_attack(
    model_orig: torch.nn.Module,
    model_unlearned: torch.nn.Module,
    data_orig: Data,
    data_unlearned: Data,
    hub_indices: List[int],
    control_indices: List[int],
    metric: str = 'l2'
) -> Dict[str, any]:
    """
    Run complete Hub-Ripple MIA attack across multiple trials.

    Args:
        model_orig: Original model (with target)
        model_unlearned: Unlearned model (without target)
        data_orig: Original graph data
        data_unlearned: Unlearned graph data
        hub_indices: List of hub node indices
        control_indices: List of control node indices
        metric: Distance metric for drift

    Returns:
        Dictionary with attack results including AUC, TPR@FPR, SNR

    Example:
        >>> results = hub_ripple_attack(model_orig, model_unlearned,
        ...                              data_orig, data_unlearned,
        ...                              hub_indices, control_indices)
        >>> print(f"AUC: {results['auc']:.4f}")
    """
    assert len(hub_indices) == len(control_indices), \
        "Hub and control indices must have same length"

    # Collect drift measurements
    y_true = []  # 1 for hub (positive), 0 for control (negative)
    y_scores = []  # Drift values
    hub_drifts = []
    control_drifts = []

    for hub_idx, control_idx in zip(hub_indices, control_indices):
        result = hub_ripple_single_trial(
            model_orig, model_unlearned,
            data_orig, data_unlearned,
            hub_idx, control_idx, metric
        )

        # Hub sample (positive class)
        y_true.append(1)
        y_scores.append(result['hub_drift'])
        hub_drifts.append(result['hub_drift'])

        # Control sample (negative class)
        y_true.append(0)
        y_scores.append(result['control_drift'])
        control_drifts.append(result['control_drift'])

    # Compute metrics
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Compute TPR at specific FPR levels
    tpr_at_fpr = {}
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.where(fpr <= target_fpr)[0]
        if len(idx) > 0:
            tpr_at_fpr[f'tpr@fpr={target_fpr}'] = tpr[idx[-1]]
        else:
            tpr_at_fpr[f'tpr@fpr={target_fpr}'] = 0.0

    # Signal-to-noise ratio
    mean_hub_drift = np.mean(hub_drifts)
    mean_control_drift = np.mean(control_drifts)
    snr = mean_hub_drift / (mean_control_drift + 1e-9)

    return {
        'auc': auc,
        'mean_hub_drift': mean_hub_drift,
        'mean_control_drift': mean_control_drift,
        'snr': snr,
        'hub_drifts': hub_drifts,
        'control_drifts': control_drifts,
        'y_true': y_true,
        'y_scores': y_scores,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        **tpr_at_fpr
    }


def interpret_attack_results(results: Dict) -> str:
    """
    Generate human-readable interpretation of attack results.

    Args:
        results: Results from hub_ripple_attack()

    Returns:
        Formatted interpretation string

    Example:
        >>> interpretation = interpret_attack_results(results)
        >>> print(interpretation)
    """
    auc = results['auc']
    snr = results['snr']

    lines = []
    lines.append("=" * 70)
    lines.append("Hub-Ripple MIA Results")
    lines.append("=" * 70)
    lines.append(f"AUC Score: {auc:.4f}")
    lines.append(f"Mean Hub Drift: {results['mean_hub_drift']:.6f}")
    lines.append(f"Mean Control Drift: {results['mean_control_drift']:.6f}")
    lines.append(f"Signal-to-Noise Ratio: {snr:.2f}x")
    lines.append("")

    # TPR at FPR thresholds
    for key in ['tpr@fpr=0.01', 'tpr@fpr=0.05', 'tpr@fpr=0.10']:
        if key in results:
            fpr_val = key.split('=')[1]
            lines.append(f"TPR @ FPR={fpr_val}: {results[key]:.4f}")

    lines.append("")
    lines.append("-" * 70)

    # Vulnerability assessment
    if auc >= 0.90:
        verdict = "HIGHLY VULNERABLE"
        explanation = "Hub-Ripple attack can reliably detect unlearning events."
    elif auc >= 0.70:
        verdict = "MODERATELY VULNERABLE"
        explanation = "Attack shows moderate success. Some protection exists."
    else:
        verdict = "RESISTANT"
        explanation = "Attack largely fails. Good privacy protection."

    lines.append(f"Verdict: {verdict}")
    lines.append(explanation)
    lines.append("=" * 70)

    return "\n".join(lines)
