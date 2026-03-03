"""Part 4a/4b: RL controller — 정책 네트워크, 리플레이 메모리, 보상 계산.

정책 네트워크:
- 2-layer MLP, hidden=32, ReLU
- 입력: 24차원 상태 벡터 (state.py에서 구성)
- 출력: 3개 액션 logits → {-beta, 0, +beta}

학습 알고리즘:
- REINFORCE with baseline (외부 RL 라이브러리 미사용, 직접 구현)
- Replay memory (capacity=1000)
"""

import random
from collections import deque

import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# 정책 네트워크
# ---------------------------------------------------------------------------

class ALAPolicy(nn.Module):
    """ALA RL controller 정책 네트워크.

    2-layer MLP: state(24d) → 32 → ReLU → 32 → ReLU → 3 (action logits).
    액션 공간: {-beta, 0, +beta} → 인덱스 {0, 1, 2}.
    """

    def __init__(self, state_dim: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """상태를 받아 액션 logits 반환."""
        return self.net(state)

    def select_action(
        self, state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """정책에서 액션 샘플링.

        Args:
            state: (B, 24) 또는 (24,) 상태 텐서.

        Returns:
            action_indices: (B,) 샘플된 액션 인덱스 (0, 1, 2).
            log_probs: (B,) 샘플된 액션의 log-probability.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs


# ---------------------------------------------------------------------------
# 리플레이 메모리
# ---------------------------------------------------------------------------

class ReplayMemory:
    """(states, actions, log_probs, reward) 튜플을 저장하는 리플레이 버퍼.

    FIFO 방식 (deque), 최대 용량 초과 시 가장 오래된 항목 제거.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self.buffer: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = (
            deque(maxlen=capacity)
        )

    def push(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        reward: float,
    ) -> None:
        """경험 저장 (CPU로 detach하여 GPU 메모리 절약)."""
        self.buffer.append((
            states.detach().cpu(),
            actions.detach().cpu(),
            log_probs.detach().cpu(),
            reward,
        ))

    def sample(
        self, batch_size: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:
        """랜덤 미니배치 샘플링."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 보상 계산 (논문 Eq.5)
# ---------------------------------------------------------------------------

def compute_reward(M_old: float, M_new: float) -> float:
    """보상 계산: r_t = sign(M_{t-1} - M_t).

    M = 검증 오류율. 오류가 줄었으면 +1, 늘었으면 -1, 같으면 0.
    """
    diff = M_old - M_new
    if diff > 0:
        return 1.0
    elif diff < 0:
        return -1.0
    else:
        return 0.0
