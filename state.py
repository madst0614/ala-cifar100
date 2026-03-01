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
    confusion = torch.zeros(num_classes, num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            neg_log_probs = -torch.log(probs + 1e-8)  # (B, C)

            # Vectorized accumulation via index_add
            confusion.index_add_(0, targets, neg_log_probs)
            class_counts.scatter_add_(
                0, targets, torch.ones(targets.size(0), device=device),
            )

    # Normalize by class counts
    class_counts = class_counts.clamp(min=1e-8)
    confusion = confusion / class_counts.unsqueeze(1)

    return confusion


def get_pair_indices(num_classes: int = 100) -> list[tuple[int, int]]:
    """Return upper-triangular class pair indices (i, j) where i < j."""
    return [(i, j) for i in range(num_classes) for j in range(i + 1, num_classes)]


def get_pair_indices_tensor(
    num_classes: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pair_i, pair_j) as tensors for vectorized indexing."""
    indices = get_pair_indices(num_classes)
    pair_i = torch.tensor([p[0] for p in indices], device=device)
    pair_j = torch.tensor([p[1] for p in indices], device=device)
    return pair_i, pair_j


def construct_states(
    confusion_history: list[torch.Tensor],
    phi: torch.Tensor,
    progress: float,
    num_classes: int = 100,
    pair_i: torch.Tensor | None = None,
    pair_j: torch.Tensor | None = None,
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
    device = phi.device
    num_pairs = num_classes * (num_classes - 1) // 2

    # Build pair index tensors if not provided
    if pair_i is None or pair_j is None:
        pair_i, pair_j = get_pair_indices_tensor(num_classes, device)

    T = len(confusion_history)
    max_T = 10
    valid_steps = min(T, max_T)

    # Extract time series via advanced indexing: (max_T, num_pairs, 2)
    ts_data = torch.zeros(max_T, num_pairs, 2, device=device)
    for t_idx in range(valid_steps):
        actual_idx = T - valid_steps + t_idx
        C = confusion_history[actual_idx]
        ts_data[t_idx, :, 0] = C[pair_i, pair_j]
        ts_data[t_idx, :, 1] = C[pair_j, pair_i]

    # 1. Time series flattened: (num_pairs, 20)
    ts_flat = ts_data.permute(1, 0, 2).reshape(num_pairs, max_T * 2)

    # 2. Relative change: (num_pairs, 2)
    if T > 1:
        current = ts_data[valid_steps - 1]               # (num_pairs, 2)
        mean_vals = ts_data[:valid_steps].mean(dim=0)     # (num_pairs, 2)
        relative_change = (current - mean_vals) / (mean_vals + 1e-8)
    else:
        relative_change = torch.zeros(num_pairs, 2, device=device)

    # 3. Current phi values via advanced indexing: (num_pairs, 1)
    phi_vals = phi[pair_i, pair_j].unsqueeze(1)

    # 4. Progress: (num_pairs, 1)
    progress_vals = torch.full((num_pairs, 1), progress, device=device)

    # Concatenate: 20 + 2 + 1 + 1 = 24
    states = torch.cat([ts_flat, relative_change, phi_vals, progress_vals], dim=1)

    return states
