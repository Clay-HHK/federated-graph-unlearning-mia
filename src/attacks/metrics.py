"""
Attack evaluation metrics.

This module provides comprehensive metrics for evaluating
membership inference attacks on graph unlearning.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from typing import Dict, List, Tuple


def compute_attack_metrics(
    y_true: List[int],
    y_scores: List[float]
) -> Dict[str, float]:
    """
    Compute comprehensive attack evaluation metrics.

    Args:
        y_true: True labels (1 for member/hub, 0 for non-member/control)
        y_scores: Attack scores (higher = more likely member)

    Returns:
        Dictionary with all metrics

    Example:
        >>> metrics = compute_attack_metrics(y_true, y_scores)
        >>> print(f"AUC: {metrics['auc']:.4f}")
    """
    # ROC AUC
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # TPR at fixed FPR levels
    tpr_at_fpr = {}
    for target_fpr in [0.001, 0.01, 0.05, 0.10]:
        idx = np.where(fpr <= target_fpr)[0]
        if len(idx) > 0:
            tpr_at_fpr[f'tpr@fpr={target_fpr}'] = float(tpr[idx[-1]])
        else:
            tpr_at_fpr[f'tpr@fpr={target_fpr}'] = 0.0

    # Precision-recall
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = np.trapz(recall, precision)

    # Advantage (TPR - FPR at optimal threshold)
    optimal_idx = np.argmax(tpr - fpr)
    advantage = float(tpr[optimal_idx] - fpr[optimal_idx])

    return {
        'auc': float(auc),
        'pr_auc': float(pr_auc),
        'advantage': advantage,
        **tpr_at_fpr
    }


def compute_signal_to_noise(
    positive_scores: List[float],
    negative_scores: List[float]
) -> Dict[str, float]:
    """
    Compute signal-to-noise ratio and separation metrics.

    Args:
        positive_scores: Scores for positive class (hub/member)
        negative_scores: Scores for negative class (control/non-member)

    Returns:
        Dictionary with SNR and separation metrics

    Example:
        >>> snr_metrics = compute_signal_to_noise(hub_drifts, control_drifts)
        >>> print(f"SNR: {snr_metrics['snr']:.2f}x")
    """
    pos_mean = np.mean(positive_scores)
    pos_std = np.std(positive_scores)
    neg_mean = np.mean(negative_scores)
    neg_std = np.std(negative_scores)

    # Signal-to-noise ratio
    snr = pos_mean / (neg_mean + 1e-9)

    # Separation (Cohen's d)
    pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
    cohens_d = (pos_mean - neg_mean) / (pooled_std + 1e-9)

    # Overlap coefficient
    pos_min, pos_max = np.min(positive_scores), np.max(positive_scores)
    neg_min, neg_max = np.min(negative_scores), np.max(negative_scores)

    overlap_start = max(pos_min, neg_min)
    overlap_end = min(pos_max, neg_max)
    overlap = max(0, overlap_end - overlap_start)
    total_range = max(pos_max, neg_max) - min(pos_min, neg_min)
    overlap_ratio = overlap / (total_range + 1e-9)

    return {
        'snr': float(snr),
        'cohens_d': float(cohens_d),
        'overlap_ratio': float(overlap_ratio),
        'positive_mean': float(pos_mean),
        'positive_std': float(pos_std),
        'negative_mean': float(neg_mean),
        'negative_std': float(neg_std)
    }


def compute_statistical_significance(
    positive_scores: List[float],
    negative_scores: List[float]
) -> Dict[str, float]:
    """
    Test statistical significance of score differences.

    Uses Wilcoxon rank-sum test (non-parametric).

    Args:
        positive_scores: Scores for positive class
        negative_scores: Scores for negative class

    Returns:
        Dictionary with test statistics and p-value

    Example:
        >>> sig = compute_statistical_significance(hub_drifts, control_drifts)
        >>> print(f"p-value: {sig['p_value']:.2e}")
    """
    from scipy.stats import ranksums

    statistic, p_value = ranksums(positive_scores, negative_scores)

    return {
        'test': 'wilcoxon_rank_sum',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def compute_confidence_interval(
    scores: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for scores.

    Uses bootstrap resampling.

    Args:
        scores: Score values
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)

    Example:
        >>> mean, lower, upper = compute_confidence_interval(hub_drifts)
        >>> print(f"Mean: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
    """
    n_bootstrap = 1000
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    mean = np.mean(scores)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(mean), float(lower), float(upper)


def summarize_attack_performance(
    y_true: List[int],
    y_scores: List[float],
    positive_label: str = "Hub",
    negative_label: str = "Control"
) -> str:
    """
    Generate comprehensive attack performance summary.

    Args:
        y_true: True labels
        y_scores: Attack scores
        positive_label: Name for positive class (default: "Hub")
        negative_label: Name for negative class (default: "Control")

    Returns:
        Formatted summary string

    Example:
        >>> summary = summarize_attack_performance(y_true, y_scores)
        >>> print(summary)
    """
    # Separate scores by class
    pos_scores = [s for y, s in zip(y_true, y_scores) if y == 1]
    neg_scores = [s for y, s in zip(y_true, y_scores) if y == 0]

    # Compute metrics
    attack_metrics = compute_attack_metrics(y_true, y_scores)
    snr_metrics = compute_signal_to_noise(pos_scores, neg_scores)
    sig_test = compute_statistical_significance(pos_scores, neg_scores)

    # Format output
    lines = []
    lines.append("=" * 70)
    lines.append("Attack Performance Summary")
    lines.append("=" * 70)
    lines.append("")

    # Primary metrics
    lines.append("Primary Metrics:")
    lines.append(f"  AUC: {attack_metrics['auc']:.4f}")
    lines.append(f"  PR-AUC: {attack_metrics['pr_auc']:.4f}")
    lines.append(f"  Advantage: {attack_metrics['advantage']:.4f}")
    lines.append("")

    # TPR at FPR
    lines.append("True Positive Rate at Fixed FPR:")
    for fpr_level in [0.001, 0.01, 0.05, 0.10]:
        key = f'tpr@fpr={fpr_level}'
        if key in attack_metrics:
            lines.append(f"  TPR @ FPR={fpr_level}: {attack_metrics[key]:.4f}")
    lines.append("")

    # Signal metrics
    lines.append("Signal Metrics:")
    lines.append(f"  {positive_label} Mean: {snr_metrics['positive_mean']:.6f} "
                 f"± {snr_metrics['positive_std']:.6f}")
    lines.append(f"  {negative_label} Mean: {snr_metrics['negative_mean']:.6f} "
                 f"± {snr_metrics['negative_std']:.6f}")
    lines.append(f"  Signal-to-Noise Ratio: {snr_metrics['snr']:.2f}x")
    lines.append(f"  Cohen's d: {snr_metrics['cohens_d']:.2f}")
    lines.append(f"  Distribution Overlap: {snr_metrics['overlap_ratio']:.1%}")
    lines.append("")

    # Statistical significance
    lines.append("Statistical Significance:")
    lines.append(f"  Test: {sig_test['test']}")
    lines.append(f"  p-value: {sig_test['p_value']:.2e}")
    lines.append(f"  Significant (α=0.05): {'Yes' if sig_test['significant'] else 'No'}")
    lines.append("")

    # Interpretation
    auc = attack_metrics['auc']
    if auc >= 0.90:
        verdict = "HIGHLY VULNERABLE"
    elif auc >= 0.70:
        verdict = "MODERATELY VULNERABLE"
    else:
        verdict = "RESISTANT"

    lines.append("-" * 70)
    lines.append(f"Overall Verdict: {verdict}")
    lines.append("=" * 70)

    return "\n".join(lines)
