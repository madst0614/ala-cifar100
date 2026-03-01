"""Part 3: Confusion matrix computation and state representation for RL controller."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_confusion_matrix(
    model: nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute confusion matrix C where C_ij = avg negative log-prob of class j for class i samples.

    Eq.7: C_ij = sum_d -I(y_d, i) * log f^j_w(x_d) / sum_d I(y_d, i)

    Returns:
        (num_classes, num_classes) tensor.
    """
    model.eval()
    # Accumulate negative log-probs per true class
    confusion = torch.zeros(num_classes, num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            neg_log_probs = -torch.log(probs + 1e-8)  # (B, C)

            for c in range(num_classes):
                mask = targets == c
                if mask.any():
                    confusion[c] += neg_log_probs[mask].sum(dim=0)
                    class_counts[c] += mask.sum()

    # Normalize by class counts
    class_counts = class_counts.clamp(min=1e-8)
    confusion = confusion / class_counts.unsqueeze(1)

    return confusion


def get_pair_indices(num_classes: int = 100) -> list[tuple[int, int]]:
    """Return upper-triangular class pair indices (i, j) where i < j."""
    return [(i, j) for i in range(num_classes) for j in range(i + 1, num_classes)]


def construct_states(
    confusion_history: list[torch.Tensor],
    phi: torch.Tensor,
    progress: float,
    num_classes: int = 100,
) -> torch.Tensor:
    """Construct state vectors for all upper-triangular class pairs.

    State components per pair (i, j):
        1. Validation statistics time series (20d): past 10 timesteps of [C_ij, C_ji]
        2. Relative change (2d): (current - mean) / (mean + 1e-8)
        3. Current phi(i,j) value (1d)
        4. Training progress (1d)

    Returns:
        (num_pairs, 24) tensor where num_pairs = num_classes*(num_classes-1)//2 = 4950.
    """
    pair_indices = get_pair_indices(num_classes)
    num_pairs = len(pair_indices)
    device = phi.device

    # Collect time series: (T, num_pairs, 2)
    T = len(confusion_history)
    max_T = 10

    # Pre-extract [C_ij, C_ji] for all pairs across all timesteps
    ts_data = torch.zeros(max_T, num_pairs, 2, device=device)
    for t_idx in range(min(T, max_T)):
        # Use most recent entries, padded from the left with zeros
        actual_idx = T - min(T, max_T) + t_idx
        C = confusion_history[actual_idx]
        for p_idx, (i, j) in enumerate(pair_indices):
            ts_data[t_idx, p_idx, 0] = C[i, j]
            ts_data[t_idx, p_idx, 1] = C[j, i]

    # 1. Time series flattened: (num_pairs, 20)
    ts_flat = ts_data.permute(1, 0, 2).reshape(num_pairs, max_T * 2)

    # 2. Relative change: (num_pairs, 2)
    if T > 1:
        # Current values
        current = ts_data[min(T, max_T) - 1]  # (num_pairs, 2)
        # Mean of history
        valid_steps = min(T, max_T)
        mean_vals = ts_data[:valid_steps].mean(dim=0)  # (num_pairs, 2)
        relative_change = (current - mean_vals) / (mean_vals + 1e-8)
    else:
        relative_change = torch.zeros(num_pairs, 2, device=device)

    # 3. Current phi values: (num_pairs, 1)
    phi_vals = torch.zeros(num_pairs, 1, device=device)
    for p_idx, (i, j) in enumerate(pair_indices):
        phi_vals[p_idx, 0] = phi[i, j]

    # 4. Progress: (num_pairs, 1)
    progress_vals = torch.full((num_pairs, 1), progress, device=device)

    # Concatenate: 20 + 2 + 1 + 1 = 24
    states = torch.cat([ts_flat, relative_change, phi_vals, progress_vals], dim=1)

    return states
