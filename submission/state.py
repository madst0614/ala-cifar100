"""Part 3: 혼동행렬 계산 및 RL controller용 상태 표현 구성.

상태 벡터 구성 (pair (i,j) 당 24차원):
1. Validation statistics (20d): 과거 10 timestep의 [C_ij, C_ji]
2. Relative change (2d): (현재 - 이동평균) / (이동평균 + eps)
3. Current Phi_t(i,j) (1d): 현재 Phi 값
4. Normalized iteration (1d): 학습 진행률 [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 혼동행렬 (논문 Eq.7)
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    model: nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """혼동행렬 계산 (Eq.7).

    C_ij = sum_d -I(y_d, i) * log f^j_w(x_d) / sum_d I(y_d, i)
    클래스 i에 속하는 샘플들에 대해 클래스 j의 평균 negative log-probability.

    Args:
        model: 현재 모델.
        val_loader: 검증 데이터 로더.
        num_classes: 클래스 수.
        device: 연산 장치.

    Returns:
        (num_classes, num_classes) 혼동행렬 텐서.
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

            # 벡터화된 누적: 각 클래스별 neg_log_probs 합산
            confusion.index_add_(0, targets, neg_log_probs)
            class_counts.scatter_add_(
                0, targets, torch.ones(targets.size(0), device=device),
            )

    # 클래스별 샘플 수로 정규화
    class_counts = class_counts.clamp(min=1e-8)
    confusion = confusion / class_counts.unsqueeze(1)

    return confusion


# ---------------------------------------------------------------------------
# 클래스 페어 인덱스
# ---------------------------------------------------------------------------

def get_pair_indices(num_classes: int = 100) -> list[tuple[int, int]]:
    """상삼각 클래스 페어 인덱스 반환: (i, j) where i < j.

    CIFAR-100: 100*99/2 = 4950 페어.
    """
    return [(i, j) for i in range(num_classes) for j in range(i + 1, num_classes)]


def get_pair_indices_tensor(
    num_classes: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """벡터화 인덱싱용 텐서 반환: (pair_i, pair_j)."""
    indices = get_pair_indices(num_classes)
    pair_i = torch.tensor([p[0] for p in indices], device=device)
    pair_j = torch.tensor([p[1] for p in indices], device=device)
    return pair_i, pair_j


# ---------------------------------------------------------------------------
# 상태 벡터 구성
# ---------------------------------------------------------------------------

def construct_states(
    confusion_history: list[torch.Tensor],
    phi: torch.Tensor,
    progress: float,
    num_classes: int = 100,
    pair_i: torch.Tensor | None = None,
    pair_j: torch.Tensor | None = None,
) -> torch.Tensor:
    """모든 상삼각 클래스 페어에 대한 상태 벡터 구성.

    상태 = [validation_ts(20d), relative_change(2d), phi_val(1d), progress(1d)] = 24d

    Args:
        confusion_history: 과거 혼동행렬 리스트 (최대 10개 사용).
        phi: (C, C) 현재 Phi 행렬.
        progress: 학습 진행률 [0, 1].
        num_classes: 클래스 수.
        pair_i, pair_j: 미리 계산된 페어 인덱스 텐서 (성능 최적화).

    Returns:
        (num_pairs, 24) 상태 텐서.
    """
    device = phi.device
    num_pairs = num_classes * (num_classes - 1) // 2

    if pair_i is None or pair_j is None:
        pair_i, pair_j = get_pair_indices_tensor(num_classes, device)

    T = len(confusion_history)
    max_T = 10
    valid_steps = min(T, max_T)

    # 시계열 데이터 추출: (max_T, num_pairs, 2)
    ts_data = torch.zeros(max_T, num_pairs, 2, device=device)
    for t_idx in range(valid_steps):
        actual_idx = T - valid_steps + t_idx
        C = confusion_history[actual_idx]
        ts_data[t_idx, :, 0] = C[pair_i, pair_j]
        ts_data[t_idx, :, 1] = C[pair_j, pair_i]

    # 1. 시계열 평탄화: (num_pairs, 20)
    ts_flat = ts_data.permute(1, 0, 2).reshape(num_pairs, max_T * 2)

    # 2. 상대 변화량: (num_pairs, 2)
    if T > 1:
        current = ts_data[valid_steps - 1]                # (num_pairs, 2)
        mean_vals = ts_data[:valid_steps].mean(dim=0)      # (num_pairs, 2)
        relative_change = (current - mean_vals) / (mean_vals + 1e-8)
    else:
        relative_change = torch.zeros(num_pairs, 2, device=device)

    # 3. 현재 Phi 값: (num_pairs, 1)
    phi_vals = phi[pair_i, pair_j].unsqueeze(1)

    # 4. 학습 진행률: (num_pairs, 1)
    progress_vals = torch.full((num_pairs, 1), progress, device=device)

    # 연결: 20 + 2 + 1 + 1 = 24
    states = torch.cat([ts_flat, relative_change, phi_vals, progress_vals], dim=1)

    return states
