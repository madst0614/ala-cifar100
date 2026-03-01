"""Part 4a/4b: Policy network, replay memory, and reward computation for ALA."""

import random
from collections import deque

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ALAPolicy(nn.Module):
    """Policy network for the ALA RL controller.

    2-layer MLP with hidden size 32.
    Maps state vectors to action logits over {-beta, 0, +beta}.
    Default PyTorch initialization (no custom bias init).
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
        return self.net(state)

    def select_action(
        self, state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the policy.

        Returns:
            action_indices: (B,) sampled action indices (0, 1, or 2).
            log_probs: (B,) log-probabilities of sampled actions.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs


class ReplayMemory:
    """Replay buffer using deque for (states, actions, log_probs, reward) tuples."""

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
        self.buffer.append((
            states.detach().cpu(),
            actions.detach().cpu(),
            log_probs.detach().cpu(),
            reward,
        ))

    def sample(
        self, batch_size: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


def compute_reward(M_old: float, M_new: float) -> float:
    """Compute reward (Eq.5): r_t = sign(M_old - M_new)."""
    diff = M_old - M_new
    if diff > 0:
        return 1.0
    elif diff < 0:
        return -1.0
    else:
        return 0.0
