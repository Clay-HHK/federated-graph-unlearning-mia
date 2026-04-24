"""
Differential Privacy utilities for federated GNN training.

Implements manual DP-SGD (gradient clipping + Gaussian noise) for
PyG-based GNN models, which are incompatible with Opacus due to
sparse message-passing operations.

Reference: Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
"""

import math
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DPConfig:
    """Differential privacy configuration.

    Args:
        epsilon: Privacy budget (lower = more private)
        delta: Failure probability
        max_grad_norm: Gradient clipping bound C
    """

    epsilon: float
    delta: float = 1e-5
    max_grad_norm: float = 1.0

    @property
    def noise_multiplier(self) -> float:
        """Compute sigma from epsilon via Gaussian mechanism."""
        if self.epsilon == float('inf'):
            return 0.0
        return math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be positive, got {self.max_grad_norm}"
            )


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Clip all parameter gradients to max_norm (global L2 norm).

    Args:
        model: Neural network model
        max_norm: Maximum gradient norm

    Returns:
        Original gradient norm before clipping
    """
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(total_norm)


def add_gaussian_noise(
    model: nn.Module,
    noise_multiplier: float,
    max_grad_norm: float,
) -> None:
    """
    Add calibrated Gaussian noise to all parameter gradients.

    The noise scale is sigma * C where sigma is the noise multiplier
    and C is the gradient clipping bound.

    Args:
        model: Neural network model (gradients will be modified in-place)
        noise_multiplier: sigma = sqrt(2 * ln(1.25/delta)) / epsilon
        max_grad_norm: Gradient clipping bound C
    """
    if noise_multiplier == 0.0:
        return

    noise_std = noise_multiplier * max_grad_norm

    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_std
            param.grad.add_(noise)


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the total L2 norm of all parameter gradients.

    Useful for calibrating max_grad_norm before running DP-SGD.

    Args:
        model: Neural network model

    Returns:
        Total gradient L2 norm
    """
    total_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm_sq += param.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm_sq)
