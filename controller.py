"""Part 4a/4b: Policy network, replay memory, and reward computation for ALA."""

import random

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ALAPolicy(nn.Module):
    """Policy network for the ALA RL controller.

    Maps state vectors to action logits over {-beta, 0, +beta}.
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
        """Compute action logits.

        Args:
            state: (B, state_dim) or (state_dim,) tensor.

        Returns:
            (B, 3) action logits.
        """
        return self.net(state)

    def select_action(
        self, state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the policy.

        Args:
            state: (B, state_dim) or (state_dim,) tensor.

        Returns:
            action_indices: (B,) sampled action indices (0, 1, or 2).
            log_probs: (B,) log-probabilities of sampled actions.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.forward(state)  # (B, 3)
        dist = Categorical(logits=logits)
        actions = dist.sample()       # (B,)
        log_probs = dist.log_prob(actions)  # (B,)
        return actions, log_probs


class ReplayMemory:
    """Simple replay buffer for storing (states, actions, log_probs, reward) tuples."""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self.buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []

    def push(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        reward: float,
    ) -> None:
        """Store a transition.

        Args:
            states: (num_pairs, state_dim).
            actions: (num_pairs,) action indices.
            log_probs: (num_pairs,) log-probabilities.
            reward: scalar reward shared by all pairs.
        """
        self.buffer.append((
            states.detach().cpu(),
            actions.detach().cpu(),
            log_probs.detach().cpu(),
            reward,
        ))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(
        self, batch_size: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


def compute_discounted_metric(metrics: list[float], gamma: float = 0.9) -> float:
    """Compute discounted metric (Eq.4).

    M_{t+1} = sum_{j=1}^{K} gamma^{K-j} * M_j

    Args:
        metrics: K validation error values.
        gamma: discount factor.

    Returns:
        Discounted sum.
    """
    K = len(metrics)
    result = 0.0
    for j in range(K):
        result += (gamma ** (K - 1 - j)) * metrics[j]
    return result


def compute_reward(M_old: float, M_new: float) -> float:
    """Compute reward (Eq.5): r_t = sign(M_old - M_new).

    M is classification error, so a decrease yields +1.

    Returns:
        +1.0 or -1.0.
    """
    if M_old > M_new:
        return 1.0
    else:
        return -1.0
