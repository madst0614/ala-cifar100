"""Part 2: 논문 Eq.6 — 파라메트릭 적응형 손실 함수.

l_Phi(f_w(x), y) = -sigma(y^T Phi log f_w(y|x))

핵심 설계 결정:
- sigma(z / sqrt(num_classes)) 정규화 적용
  CIFAR-100에서 num_classes=100이면 inner product y^T Phi log f_w의 절대값이 크다.
  이 값이 그대로 sigmoid에 들어가면 saturation 발생 → gradient가 거의 0.
  sqrt(100)=10으로 나눠서 sigmoid의 선형 영역을 활용, gradient flow 유지.
- Phi는 gradient descent가 아닌 RL controller에 의해 업데이트
- 대각선은 항상 1.0 유지, off-diagonal만 RL이 조정
- symmetry constraint: Phi(i,j) = Phi(j,i) 강제
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLoss(nn.Module):
    """적응형 손실 함수 (논문 Eq.6).

    Phi는 클래스 관계 행렬로, RL controller가 update_phi()를 통해 조정.
    gradient descent로는 학습되지 않음 (requires_grad=False).

    Args:
        num_classes: 클래스 수 (CIFAR-100 = 100).
        clamp_range: off-diagonal Phi 값의 허용 범위.
    """

    def __init__(self, num_classes: int = 100, clamp_range: tuple = (-1.0, 1.0)) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.clamp_range = clamp_range
        # Phi_0 = I (항등행렬) — RL agent가 off-diagonal을 조정
        self.phi = nn.Parameter(torch.eye(num_classes), requires_grad=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """적응형 손실 계산 (Eq.6 + sqrt normalization).

        l = -sigma(y^T Phi log f_w(y|x) / sqrt(C))

        Args:
            logits: (B, C) 모델의 raw output (softmax 이전).
            targets: (B,) 정수 클래스 레이블.

        Returns:
            스칼라 손실 (배치 평균).
        """
        log_probs = F.log_softmax(logits, dim=1)              # (B, C)
        y = F.one_hot(targets, self.num_classes).float()       # (B, C)
        weighted = y @ self.phi                                 # (B, C)
        inner = (weighted * log_probs).sum(dim=1)              # (B,)

        # sqrt normalization: sigmoid saturation 방지
        inner = inner / math.sqrt(self.num_classes)

        sig = torch.sigmoid(inner)                              # (B,)
        return (-sig).mean()

    def update_phi(self, delta_phi: torch.Tensor) -> None:
        """Phi에 delta를 더한 후 제약 조건 적용.

        순서가 중요:
        1. delta 적용
        2. 대각선 1.0 복원 (RL은 off-diagonal만 조정)
        3. 대칭화: (Phi + Phi^T) / 2 — 논문 Section 4에서 class pair (i,j)에 대해
           동일 controller가 Phi(i,j)와 Phi(j,i)를 같은 값으로 업데이트한다고 명시
        4. 대각선 다시 1.0 복원 (대칭화로 깨질 수 있음)
        5. off-diagonal clamp

        Args:
            delta_phi: (C, C) 업데이트 텐서.
        """
        self.phi.data += delta_phi
        self.phi.data.fill_diagonal_(1.0)
        # 대칭 제약: Phi(i,j) = Phi(j,i)
        self.phi.data = (self.phi.data + self.phi.data.T) / 2
        self.phi.data.fill_diagonal_(1.0)
        # off-diagonal clamp
        diag_mask = torch.eye(self.num_classes, device=self.phi.device, dtype=torch.bool)
        lo, hi = self.clamp_range
        self.phi.data[~diag_mask] = self.phi.data[~diag_mask].clamp(lo, hi)

    def reset_phi(self) -> None:
        """Phi를 항등행렬로 초기화."""
        self.phi.data = torch.eye(self.num_classes, device=self.phi.device)
